using System.Runtime.Intrinsics.X86;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Convolutional;

public unsafe partial class CnnMatrix
{
    public class Im2ColParallel
    {
        private ParallelLocal _stateTemplate;
        private int _prevTaskSlot = int.MinValue;

        public ParallelLocal CreateTaskState() => _stateTemplate with { TaskSlot = Interlocked.Increment(ref _prevTaskSlot) };

        public ParallelLoopResult Invoke(CnnMatrix @this, NeuralMatrix colInput, int kernelH, int kernelW, int stride, int padding)
        {
            _stateTemplate.@this = @this;
            _stateTemplate.kernelH = kernelH;
            _stateTemplate.kernelW = kernelW;
            _stateTemplate.stride = stride;
            _stateTemplate.padding = padding;

            _stateTemplate.outH = ((@this.Height + (2 * _stateTemplate.padding) - _stateTemplate.kernelH) / _stateTemplate.stride) + 1;
            _stateTemplate.outW = ((@this.Width + (2 * _stateTemplate.padding) - _stateTemplate.kernelW) / _stateTemplate.stride) + 1;
            _stateTemplate.spatialOut = _stateTemplate.outH * _stateTemplate.outW;
            _stateTemplate.colStride = colInput.ColumnsStride;

            _stateTemplate.srcBase = @this.Pointer;
            _stateTemplate.colBase = colInput.Pointer;

            if (Interlocked.CompareExchange(ref _prevTaskSlot, -1, int.MinValue) != int.MinValue) throw new InvalidOperationException();
            try
            {
                var res = Parallel.For(
                    fromInclusive: 0,
                    toExclusive: @this.Batch,
                    localInit: CreateTaskState,
                    body: ForBody,
                    localFinally: ForFinally
                );

                return res;
            }
            finally
            {
                Volatile.Write(ref _prevTaskSlot, int.MinValue);
            }
        }

        public static ParallelLocal ForBody(int b, ParallelLoopState plel, ParallelLocal state)
        {
            for (int c = 0; c < state.@this.Channels; ++c) state.Invoke(b, c);
            return state;
        }

        public static void ForFinally(ParallelLocal state)
        {

        }

        public struct ParallelLocal
        {
            public int TaskSlot;
            public CnnMatrix @this;
            public int kernelH;
            public int kernelW;
            public int stride;
            public int padding;
            public int outH;
            public int outW;
            public int spatialOut;
            public int colStride;
            public float* srcBase;
            public float* colBase;

            public readonly void Invoke(int b, int c)
            {
                float* srcChannel = srcBase + (long)((b * @this.Channels) + c) * (@this.Height * @this.Width);

                for (int ky = 0; ky < kernelH; ++ky)
                {
                    for (int kx = 0; kx < kernelW; ++kx)
                    {
                        int channelKernelOffset = (c * kernelH + ky) * kernelW + kx;

                        for (int oh = 0; oh < outH; ++oh)
                        {
                            int ih = (oh * stride) - padding + ky;
                            bool hInBounds = (uint)ih < (uint)@this.Height;

                            float* colRowBase = colBase + ((long)((b * spatialOut) + (oh * outW)) * colStride) + channelKernelOffset;

                            if (!hInBounds)
                            {
                                for (int ow2 = 0; ow2 < outW; ow2++)
                                {
                                    colRowBase[ow2 * colStride] = 0.0f;
                                }
                                continue;
                            }

                            int ow = 0;
                            if (padding == 0)
                            {
                                int srcYOffset = ih * @this.Width;

                                if (stride == 1)
                                {
                                    if (Avx512F.IsSupported)
                                    {
                                        if ((outW - ow) >= 16)
                                        {
                                            int vecLimit = outW - 15;
                                            for (; ow < vecLimit; ow += 16)
                                            {
                                                int iw = ow + kx;
                                                var vData = Avx512F.LoadVector512(srcChannel + srcYOffset + iw);

                                                for (int i = 0; i < 16; i++)
                                                {
                                                    colRowBase[(ow + i) * colStride] = vData[i];
                                                }
                                            }
                                        }
                                    }
                                    else if (Avx2.IsSupported)
                                    {
                                        if ((outW - ow) >= 8)
                                        {
                                            int vecLimit = outW - 7;
                                            for (; ow < vecLimit; ow += 8)
                                            {
                                                int iw = ow + kx;
                                                var vData = Avx2.LoadVector256(srcChannel + srcYOffset + iw);

                                                for (int i = 0; i < 8; i++)
                                                {
                                                    colRowBase[(ow + i) * colStride] = vData[i];
                                                }
                                            }
                                        }
                                    }
                                }


                                for (; ow < outW; ow++)
                                {
                                    int iw = ow * stride + kx;
                                    colRowBase[ow * colStride] = srcChannel[srcYOffset + iw];
                                }
                            }
                            else
                            {
                                for (; ow < outW; ow++)
                                {
                                    int iw = (ow * stride) - padding + kx;
                                    colRowBase[ow * colStride] = (iw < @this.Width) ? srcChannel[ih * @this.Width + iw] : 0.0f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}
