using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Convolutional;

public unsafe partial class CnnMatrix
{
    public class Col2ImParallel
    {
        private const int kernelW = 3;
        private const int kernelH = 3;
        private const int kernelSpatial = kernelH * kernelW; // 9

        private ParallelLocal _stateTemplate;
        private int _prevTaskSlot = int.MinValue;

        public ParallelLocal CreateTaskState() => _stateTemplate with { TaskSlot = Interlocked.Increment(ref _prevTaskSlot) };

        public void Invoke(CnnMatrix @this, NeuralMatrix colGradients)
        {
            _stateTemplate.@this = @this;
            _stateTemplate.colGradients = colGradients;
            _stateTemplate.paddedH = @this.Height + 2;
            _stateTemplate.paddedW = @this.Width + 2;

            using var paddedGrad = GetOrCreate(@this.Batch, @this.Channels, _stateTemplate.paddedH, _stateTemplate.paddedW);

            _stateTemplate.colPtr = colGradients.Pointer;
            _stateTemplate.gradPtr = paddedGrad.Pointer;

            _stateTemplate.colStride = colGradients.ColumnsStride;
            _stateTemplate.colAlignGap = colGradients.ColumnsStride - colGradients.UsedColumns;
            _stateTemplate.colYStep = _stateTemplate.colStride * @this.Width;

            _stateTemplate.padW = _stateTemplate.paddedW;
            _stateTemplate.padWH = _stateTemplate.paddedH * _stateTemplate.padW;
            _stateTemplate.padWHC = @this.Channels * _stateTemplate.padWH;

            _stateTemplate.gapW = paddedGrad.StrideH - _stateTemplate.padW;
            Debug.Assert(_stateTemplate.gapW is 0);

            _stateTemplate.baseDstPtr = @this.Pointer;
            _stateTemplate.basePaddedGradPtr = paddedGrad.Pointer;

            // paddedGrad.Clear();

            if (Interlocked.CompareExchange(ref _prevTaskSlot, -1, int.MinValue) != int.MinValue) throw new InvalidOperationException();

            try
            {
                Parallel.For(
                    fromInclusive: 0,
                    toExclusive: @this.Batch,
                    localInit: CreateTaskState,
                    body: ForBody_Step1,
                    localFinally: ForFinally
                );
                Parallel.For(
                    fromInclusive: 0,
                    toExclusive: @this.Batch,
                    localInit: CreateTaskState,
                    body: ForBody_Step2,
                    localFinally: ForFinally
                );
            }
            finally
            {
                Volatile.Write(ref _prevTaskSlot, int.MinValue);
            }
        }

        public static ParallelLocal ForBody_Step1(int index, ParallelLoopState plel, ParallelLocal state)
        {
            state.Invoke_Step1(index);
            return state;
        }

        public static ParallelLocal ForBody_Step2(int index, ParallelLoopState plel, ParallelLocal state)
        {
            state.Invoke_Step2(index);
            return state;
        }

        public static void ForFinally(ParallelLocal state) { }

        public struct ParallelLocal
        {
            public int TaskSlot;
            public CnnMatrix @this;
            public NeuralMatrix colGradients;
            public int paddedH;
            public int paddedW;
            public float* colPtr;
            public int colStride;
            public int colAlignGap;
            public int colYStep;
            public float* gradPtr;
            public long padW;
            public long padWH;
            public long padWHC;
            public long gapW;

            public float* baseDstPtr;
            public float* basePaddedGradPtr;

            public readonly void Invoke_Step1(long b)
            {
                float* src = colPtr + (b * @this.StrideC * colStride);

                float* gA = gradPtr + (b * padWHC);
                float* gB = gA + padW;
                float* gC = gB + padW;

                for (long oh = 0; oh < @this.Height; ++oh, src += colYStep)
                {
                    for (long ow = 0; ow < @this.Width; ++ow, src += colStride)
                    {
                        float* dstA = gA + ow;
                        float* dstB = gB + ow;
                        float* dstC = gC + ow;

                        for (long c = 0; c < @this.Channels; c++, src += kernelSpatial)
                        {
                            dstA[0] += src[0]; dstA[1] += src[1]; dstA[2] += src[2];
                            dstB[0] += src[3]; dstB[1] += src[4]; dstB[2] += src[5];
                            dstC[0] += src[6]; dstC[1] += src[7]; dstC[2] += src[8];

                            dstA += padWH;
                            dstB += padWH;
                            dstC += padWH;
                        }
                    }

                    Debug.Assert(gapW is 0);
                    // dstA += gapW;
                    // dstB += gapW;
                    // dstC += gapW;
                }
            }

            public readonly void Invoke_Step2(long b)
            {
                (long C, long H, long W) dstPad = (@this.StrideC, @this.StrideH, @this.StrideW);

                long batchSrcOffset = b * padWHC;
                long batchDstOffset = b * dstPad.C;

                for (long c = 0; c < @this.Channels; c++)
                {
                    float* srcChannel = basePaddedGradPtr + batchSrcOffset + (c * padWH) + padW;
                    float* dstChannel = baseDstPtr + batchDstOffset + (c * dstPad.H);

                    for (long y = 0; y < @this.Height; y++)
                    {
                        float* srcPtr = srcChannel + (y * padW) + 1;
                        float* dstPtr = dstChannel + (y * dstPad.W);
                        NativeMemory.Copy(srcPtr, dstPtr, (uint)(@this.Width * sizeof(float)));
                    }
                }
            }
        }
    }

}
