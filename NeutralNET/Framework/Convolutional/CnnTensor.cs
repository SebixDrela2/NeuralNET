using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Convolutional;

/// <summary>
/// Superoptimized 4D tensor (Batch, Channels, Height, Width) with zero-alloc L1-cached memory pooling.
/// </summary>
///
public unsafe class CnnMatrix : CriticalFinalizerObject, IDisposable
{
    public const int Alignment = SIMD.AlignSize;
    private const int ByteAlignment = SIMD.ByteAlignSize;

    public AllocationHandle MemoryHandle;
    public float* Pointer { [MethodImpl(Inline)] get => (float*)MemoryHandle.Pointer; }
    public int Batch { get; set; }
    public int Channels;
    public int Height;
    public int Width;
    public int UnsafeSize;
    public bool ReadOnly;

    public string? DisplayName { get; set; }

    public int StrideW { [MethodImpl(Inline)] get => 1; }
    public int StrideH { [MethodImpl(Inline)] get => Width; }
    public int StrideC { [MethodImpl(Inline)] get => Width * Height; }
    public int StrideN { [MethodImpl(Inline)] get => Width * Height * Channels; }

    private bool _inUse = true;
    private bool _isDisposing = false;

    public static CnnMatrix GetOrCreate(int batch, int channels, int height, int width, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
        => new(batch, channels, height, width, fp, ln);

    private CnnMatrix(int batch, int channels, int height, int width, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0, bool readOnly = false)
    {
        Batch = batch;
        Channels = channels;
        Height = height;
        Width = width;
        UnsafeSize = batch * channels * height * width;

        MemoryHandle = NeuralMemoryPool.Rent<float>(UnsafeSize);

        _inUse = true;
        Clear();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetIndex(int batch, int channel, int y, int x)
    {
        EnsureNotDisposed();
        return (batch * StrideN) + (channel * StrideC) + (y * StrideH) + x;
    }

    public ref float this[int batch, int channel, int y, int x]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            EnsureNotDisposed();
            return ref Pointer[GetIndex(batch, channel, y, x)];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetChannelPointer(int batch, int channel)
    {
        EnsureNotDisposed();
        return Pointer + (batch * StrideN) + (channel * StrideC);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetRowPointer(int batch, int channel, int y)
    {
        EnsureNotDisposed();
        return Pointer + (batch * StrideN) + (channel * StrideC) + (y * StrideH);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Clear()
    {
        EnsureNotDisposed();
        NativeMemory.Clear(Pointer, MemoryHandle.ByteSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Fill(float value)
    {
        EnsureNotDisposed();
        new Span<float>(Pointer, UnsafeSize).Fill(value);
    }

    [Conditional("DEBUG")]
    public static void AssertSameSize(CnnMatrix lhs, CnnMatrix rhs, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        AllocationHandle.AssertSameSize(lhs.MemoryHandle, rhs.MemoryHandle, fp, ln);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void CopyFrom(CnnMatrix other, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        EnsureNotDisposed();
        AssertSameSize(this, other, fp, ln);
        NativeMemory.Copy(other.Pointer, Pointer, nuint.Min(MemoryHandle.ByteSize, other.MemoryHandle.ByteSize));
    }

    public void Im2Col(NeuralMatrix colInput, int kernelH, int kernelW, int stride, int padding)
    {
        EnsureNotDisposed();

        int outH = (Height + 2 * padding - kernelH) / stride + 1;
        int outW = (Width + 2 * padding - kernelW) / stride + 1;
        int spatialOut = outH * outW;
        int colStride = colInput.ColumnsStride;

        float* srcBase = Pointer;
        float* colBase = colInput.Pointer;

        int numTasks = Batch * Channels;

        Parallel.For(0, numTasks, taskIdx =>
        {
            int b = taskIdx / Channels;
            int c = taskIdx % Channels;

            float* srcChannel = srcBase + (long)(b * Channels + c) * Height * Width;

            for (int ky = 0; ky < kernelH; ky++)
            {
                for (int kx = 0; kx < kernelW; kx++)
                {
                    int channelKernelOffset = (c * kernelH + ky) * kernelW + kx;

                    for (int oh = 0; oh < outH; oh++)
                    {
                        int ih = oh * stride - padding + ky;
                        bool hInBounds = (uint)ih < (uint)Height;

                        float* colRowBase = colBase + (long)(b * spatialOut + oh * outW) * colStride + channelKernelOffset;

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
                            int srcYOffset = ih * Width;

                            if (Avx512F.IsSupported && stride == 1 && (outW - ow) >= 16)
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
                            else if (Avx2.IsSupported && stride == 1 && (outW - ow) >= 8)
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
                                int iw = ow * stride - padding + kx;
                                colRowBase[ow * colStride] = ((uint)iw < (uint)Width) ? srcChannel[ih * Width + iw] : 0.0f;
                            }
                        }
                    }
                }
            }
        });
    }

    public void Col2Im(NeuralMatrix colGradients)
    {
        EnsureNotDisposed();

        const int kernelH = 3;
        const int kernelW = 3;
        const int kernelSpatial = kernelH * kernelW; // 9

        int paddedH = Height + 2;
        int paddedW = Width + 2;

        using var paddedGrad = GetOrCreate(Batch, Channels, paddedH, paddedW);

        float* colPtr = colGradients.Pointer;
        int colStride = colGradients.ColumnsStride;
        float* gradPtr = paddedGrad.Pointer;

        int padW = paddedW;
        long padWH = (long)paddedH * paddedW;
        long padWHC = (long)Channels * padWH;

        Unsafe.InitBlockUnaligned(gradPtr, 0, (uint)(Batch * padWHC * sizeof(float)));
        Parallel.For(0, Batch, b =>
        {
            long batchOffsetGrad = (long)b * padWHC;
            long batchPatchBase = (long)b * Height * Width;

            for (int oh = 0; oh < Height; oh++)
            {
                long patchRowBase = (batchPatchBase + (long)oh * Width) * colStride;

                for (int ow = 0; ow < Width; ow++)
                {
                    float* colRow = colPtr + patchRowBase + (long)ow * colStride;

                    for (int c = 0; c < Channels; c++)
                    {
                        long channelOffsetGrad = ((long)c * padWH) + batchOffsetGrad;
                        long channelOffsetCol = (long)c * kernelSpatial;

                        float* srcColBase = colRow + channelOffsetCol;
                        float* dstGradBase = gradPtr + channelOffsetGrad + ow;

                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            float* dstGrad = dstGradBase + ((oh + ky) * padW);
                            float* srcCol = srcColBase + (ky * kernelW);

                            dstGrad[0] += srcCol[0];
                            dstGrad[1] += srcCol[1];
                            dstGrad[2] += srcCol[2];
                        }
                    }
                }
            }
        });

        nuint rowBytes = (nuint)Width * sizeof(float);
        float* baseDstPtr = Pointer;
        float* basePaddedGradPtr = paddedGrad.Pointer;

        int numTasks = Batch * Channels;

        Parallel.For(0, numTasks, taskIdx =>
        {
            int b = taskIdx / Channels;
            int c = taskIdx % Channels;

            long batchSrcOffset = (long)b * padWHC;
            long batchDstOffset = (long)b * Channels * Height * Width;

            float* srcChannel = basePaddedGradPtr + batchSrcOffset + ((long)c * padWH);
            float* dstChannel = baseDstPtr + batchDstOffset + ((long)c * Height * Width);

            for (int y = 0; y < Height; y++)
            {
                float* srcPtr = srcChannel + ((y + 1) * padW) + 1;
                float* dstPtr = dstChannel + (y * Width);
                NativeMemory.Copy(srcPtr, dstPtr, rowBytes);
            }
        });
    }

    [OverloadResolutionPriority(1)]
    public void Dispose([CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        try
        {
            if (DisplayName is { } epic &&  epic.Contains("Conv_InputGrad"))
            {

            }

            if (_isDisposing)
            {
                throw new InvalidOperationException("ooga");
            }

            _isDisposing = true;
            EnsureNotDisposed();

            _inUse = false;

            MemoryHandle.Take().Dispose();
            GC.SuppressFinalize(this);

            _isDisposing = false;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"BORKED: {ex.Message}");
            throw;
        }
    }

    private void EnsureNotDisposed()
    {
        if (!_inUse)
        {
            throw new NotImplementedException();
        }
    }

    public void Dispose() => Dispose(ln: -1);

    ~CnnMatrix()
    {
        MemoryHandle.Free();
    }
}
