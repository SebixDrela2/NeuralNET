using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Convolutional;

/// <summary>
/// Superoptimized 4D tensor (Batch, Channels, Height, Width) with zero-alloc L1-cached memory pooling.
/// </summary>
///
public unsafe class CnnMatrix : CriticalFinalizerObject, IDisposable
{
    public static readonly ConcurrentBag<CnnMatrix>? Instances = null;
    public List<SourceLocation>? Locations = null;
    public List<SourceLocation>? DisposeLocations = null;

    // private string DebugLocations => string.Join('\n', Locations
    //     .Select(x => ("locations", x))
    //     .Concat(DisposeLocations.Select(x => ("dispose_locations", x)))
    //     .OrderBy(x => x.x.TimeStamp));

    public const int Alignment = SIMD.AlignSize;
    private const int ByteAlignment = SIMD.ByteAlignSize;

    // private static readonly ConcurrentBag<CnnMatrix> _pool = [];
    // private static readonly int CommonAllocatedLength = 134_217_728;

    public AllocationHandle MemoryHandle;
    public float* Pointer { [MethodImpl(Inline)] get => (float*)MemoryHandle.Pointer; }
    public int Batch { get; set; }
    public int Channels;
    public int Height;
    public int Width;
    public int UnsafeSize;
    public bool ReadOnly;

    public int StrideW { [MethodImpl(Inline)] get => 1; }
    public int StrideH { [MethodImpl(Inline)] get => Width; }
    public int StrideC { [MethodImpl(Inline)] get => Width * Height; }
    public int StrideN { [MethodImpl(Inline)] get => Width * Height * Channels; }

    private bool _inUse = true;
    private bool _isInit = false;

    private readonly bool _isPoolable = true;

    public static CnnMatrix Create(int batch, int channels, int height, int width, bool readOnly = false, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        var matrix = new CnnMatrix(batch, channels, height, width, isPoolable: true, fp, ln, readOnly: readOnly);

        return matrix;
    }

    public static CnnMatrix GetOrCreate(int batch, int channels, int height, int width, bool readOnly = false, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        var matrix = new CnnMatrix(batch, channels, height, width, isPoolable: true, fp, ln, readOnly: readOnly);
        // if (!_pool.TryTake(out var matrix))
        // {
        //     matrix = new CnnMatrix(batch, channels, height, width, isPoolable: true, fp, ln, readOnly: readOnly);

        //     return matrix;
        // }

        // matrix.Resize(batch, channels, height, width, fp, ln);
        return matrix;
    }

    private CnnMatrix(int batch, int channels, int height, int width, bool isPoolable, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0, bool readOnly = false)
    {
        Batch = batch;
        Channels = channels;
        Height = height;
        Width = width;
        ReadOnly = readOnly;
        UnsafeSize = batch * channels * height * width;
        _isPoolable = isPoolable;

        MemoryHandle = NeuralMemoryPool.Rent<float>(UnsafeSize);

        Locations?.Add(SourceLocation.Current(new MatrixInfo([batch, channels, height, width], UnsafeSize), fp, ln));
        // Pointer = (float*)NativeMemory.AlignedAlloc((nuint)(allocatedLength * sizeof(float)), (nuint)ByteAlignment);
        _inUse = true;
        Clear();
        Instances?.Add(this);
    }

    // private void Resize(int batch, int channels, int height, int width, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    // {
    //     if (_inUse)
    //     {
    //         throw new InvalidOperationException("cheeky");
    //     }

    //     if (_isDisposing || _isInit)
    //     {
    //         throw new InvalidOperationException("ooga init");
    //     }

    //     _isInit = true;
    //     int newSize = batch * channels * height * width;

    //     if (newSize > CommonAllocatedLength)
    //     {
    //         throw new InvalidOperationException($"Tensor size {newSize} exceeds pool buffer size {CommonAllocatedLength}.");
    //     }

    //     Locations.Add(SourceLocation.Current(new MatrixInfo([batch, channels, height, width], newSize), fp, ln));
    //     Batch = batch;
    //     Channels = channels;
    //     Height = height;
    //     Width = width;
    //     UnsafeSize = newSize;
    //     _inUse = true;
    //     _isInit = false;
    // }

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
        int paddedH = Height + 2 * padding;
        int paddedW = Width + 2 * padding;
        int outH = (paddedH - kernelH) / stride + 1;
        int outW = (paddedW - kernelW) / stride + 1;
        int patchSize = Channels * kernelH * kernelW;
        int totalPatches = Batch * outH * outW;

        float* colPtr = colInput.Pointer;
        int colStride = colInput.ColumnsStride;
        bool needsPadding = padding > 0;
        using var padded = needsPadding ? GetOrCreate(Batch, Channels, paddedH, paddedW) : null;
        if (needsPadding) padded.Clear();

        float* baseSrcPtr = Pointer;
        float* basePaddedPtr = needsPadding ? padded!.Pointer : baseSrcPtr;
        int srcStrideH = Width;
        int targetPaddedW = needsPadding ? paddedW : Width;

        if (needsPadding)
        {
            nuint rowBytes = (nuint)Width * sizeof(float);
            Parallel.For(0, Batch, b =>
            {
                long batchSrcOffset = b * Channels * Height * Width;
                long batchPadOffset = b * Channels * paddedH * paddedW;

                for (int c = 0; c < Channels; c++)
                {
                    float* srcPtr = baseSrcPtr + batchSrcOffset + c * Height * Width;
                    float* dstPtr = basePaddedPtr + batchPadOffset + c * paddedH * paddedW;

                    for (int y = 0; y < Height; y++)
                    {
                        float* srcRow = srcPtr + y * srcStrideH;
                        float* dstRow = dstPtr + (y + padding) * targetPaddedW + padding;
                        NativeMemory.Copy(srcRow, dstRow, rowBytes);
                    }
                }
            });
        }

        int spatialPadStride = paddedH * targetPaddedW;

        Parallel.For(0, Batch, b =>
        {
            int batchPatchBase = b * outH * outW;
            long batchPadOffset = b * Channels * spatialPadStride;
            long batchColOffset = (long)batchPatchBase * colStride;
            float* batchColPtr = colPtr + batchColOffset;

            for (int oh = 0; oh < outH; oh++)
            {
                int startY = oh * stride;
                int patchRowBase = oh * outW;

                for (int ow = 0; ow < outW; ow++)
                {
                    int startX = ow * stride;
                    float* dstRow = batchColPtr + (patchRowBase + ow) * colStride;
                    int colIdx = 0;

                    for (int c = 0; c < Channels; c++)
                    {
                        float* channelPaddedPtr = basePaddedPtr + batchPadOffset + c * spatialPadStride;

                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            float* srcRow = channelPaddedPtr + (startY + ky) * targetPaddedW + startX;

                            if (kernelW == 3)
                            {
                                dstRow[colIdx] = srcRow[0];
                                dstRow[colIdx + 1] = srcRow[1];
                                dstRow[colIdx + 2] = srcRow[2];
                                colIdx += 3;
                            }
                            else if (kernelW == 1)
                            {
                                dstRow[colIdx++] = srcRow[0];
                            }
                            else
                            {
                                nuint copyBytes = (nuint)kernelW * sizeof(float);
                                NativeMemory.Copy(srcRow, dstRow + colIdx, copyBytes);
                                colIdx += kernelW;
                            }
                        }
                    }
                }
            }
        });
    }

    public void Col2Im(NeuralMatrix colGradients, int kernelH, int kernelW, int stride, int padding, float scale = 1.0f)
    {
        EnsureNotDisposed();
        int paddedH = Height + 2 * padding;
        int paddedW = Width + 2 * padding;
        int outH = (paddedH - kernelH) / stride + 1;
        int outW = (paddedW - kernelW) / stride + 1;

        using var paddedGrad = GetOrCreate(Batch, Channels, paddedH, paddedW);
        float* colPtr = colGradients.Pointer;
        int colStride = colGradients.ColumnsStride;
        float* gradPtr = paddedGrad.Pointer;
        int kernelSpatial = kernelH * kernelW;

        Parallel.For(0, Batch, b =>
        {
            long batchOffsetGrad = b * paddedGrad.StrideN;
            int batchPatchBase = b * outH * outW;

            for (int oh = 0; oh < outH; oh++)
            {
                int startY = oh * stride;
                int patchRowBase = (b * outH + oh) * outW;

                for (int ow = 0; ow < outW; ow++)
                {
                    int startX = ow * stride;
                    float* colRow = colPtr + (patchRowBase + ow) * colStride;

                    for (int c = 0; c < Channels; c++)
                    {
                        long channelOffsetGrad = batchOffsetGrad + c * paddedGrad.StrideC;
                        int channelOffsetCol = c * kernelSpatial;

                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            float* dstGrad = gradPtr + channelOffsetGrad + (startY + ky) * paddedGrad.StrideH + startX;
                            float* srcCol = colRow + channelOffsetCol + ky * kernelW;

                            int kx = 0;
                            if (Avx512F.IsSupported)
                            {
                                var vScale512 = Vector512.Create(scale);
                                int vecLimit = kernelW - (kernelW % 16);
                                for (; kx < vecLimit; kx += 16)
                                {
                                    var vDst = Vector512.Load(dstGrad + kx);
                                    var vSrc = Vector512.Load(srcCol + kx);
                                    vDst = Vector512.FusedMultiplyAdd(vSrc, vScale512, vDst);
                                    vDst.Store(dstGrad + kx);
                                }
                            }
                            else if (Avx2.IsSupported)
                            {
                                var vScale256 = Vector256.Create(scale);
                                int vecLimit = kernelW - (kernelW % 8);
                                for (; kx < vecLimit; kx += 8)
                                {
                                    var vDst = Vector256.Load(dstGrad + kx);
                                    var vSrc = Vector256.Load(srcCol + kx);
                                    vDst = Vector256.FusedMultiplyAdd(vSrc, vScale256, vDst);
                                    vDst.Store(dstGrad + kx);
                                }
                            }

                            for (; kx < kernelW; kx++)
                            {
                                dstGrad[kx] += srcCol[kx] * scale;
                            }
                        }
                    }
                }
            }
        });

        nuint rowBytes = (nuint)Width * sizeof(float);
        float* baseDstPtr = Pointer;
        float* basePaddedGradPtr = paddedGrad.Pointer;

        Parallel.For(0, Batch, b =>
        {
            long batchSrcOffset = b * paddedGrad.StrideN;
            long batchDstOffset = (long)b * Channels * Height * Width;

            for (int c = 0; c < Channels; c++)
            {
                float* srcChannel = basePaddedGradPtr + batchSrcOffset + c * paddedGrad.StrideC;
                float* dstChannel = baseDstPtr + batchDstOffset + c * Height * Width;

                for (int y = 0; y < Height; y++)
                {
                    float* srcPtr = srcChannel + (y + padding) * paddedW + padding;
                    float* dstPtr = dstChannel + y * Width;
                    NativeMemory.Copy(srcPtr, dstPtr, rowBytes);
                }
            }
        });
    }

    public void Im2Col_Fast(NeuralMatrix colInput)
    {
        EnsureNotDisposed();
        const int kernelH = 3;
        const int kernelW = 3;
        const int kernelWH = kernelH * kernelW;

        int paddedH = Height + 2;
        int paddedW = Width + 2;
        int patchSize = Channels * kernelWH;
        int totalPatches = Batch * Height * Width;

        float* colPtr = colInput.Pointer;
        int colStride = colInput.ColumnsStride;
        using var padded = GetOrCreate(Batch, Channels, paddedH, paddedW);

        float* baseSrcPtr = Pointer;
        float* basePaddedPtr = padded.Pointer;
        int srcStrideH = Width;
        int targetPaddedW = paddedW;

        {
            nuint rowBytes = (nuint)Width * sizeof(float);
            for (int b = 0; b < Batch; ++b)
            {
                long batchSrcOffset = b * Channels * Height * Width;
                long batchPadOffset = b * Channels * paddedH * paddedW;

                for (int c = 0; c < Channels; c++)
                {
                    float* srcPtr = baseSrcPtr + batchSrcOffset + (c * Height * Width);
                    float* dstPtr = basePaddedPtr + batchPadOffset + (c * paddedH * paddedW);

                    for (int y = 0; y < Height; y++)
                    {
                        float* srcRow = srcPtr + y * srcStrideH;
                        float* dstRow = dstPtr + (y + 1) * targetPaddedW + 1;
                        NativeMemory.Copy(srcRow, dstRow, rowBytes);
                    }
                }
            }
        }

        int spatialPadStride = paddedH * targetPaddedW;

        for (int b = 0; b < Batch; ++b)
        {
            int batchPatchBase = b * Height * Width;
            long batchPadOffset = b * Channels * spatialPadStride;
            long batchColOffset = (long)batchPatchBase * colStride;
            float* batchColPtr = colPtr + batchColOffset;

            for (int oh = 0; oh < Height; oh++)
            {
                // int startY = oh * stride;
                int patchRowBase = oh * Width;

                for (int ow = 0; ow < Width; ow++)
                {
                    // int startX = ow * stride;
                    float* dstRow = batchColPtr + ((patchRowBase + ow) * colStride);

                    for (int c = 0; c < Channels; c++)
                    {
                        float* channelPaddedPtr = basePaddedPtr + batchPadOffset + (c * spatialPadStride);

                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            float* srcRow = channelPaddedPtr + ((oh + ky) * targetPaddedW) + ow;

                            if (kernelW == 3)
                            {
                                dstRow[0] = srcRow[0];
                                dstRow[1] = srcRow[1];
                                dstRow[2] = srcRow[2];
                                dstRow += 3;
                            }
                        }
                    }
                }
            }
        }
    }

    public void Col2Im_Fast(NeuralMatrix colGradients)
    {
        EnsureNotDisposed();
        const int kernelH = 3;
        const int kernelW = 3;
        const int kernelSpatial = kernelH * kernelW;

        int paddedH = Height + 2;
        int paddedW = Width + 2;

        using var paddedGrad = GetOrCreate(Batch, Channels, paddedH, paddedW);
        float* colPtr = colGradients.Pointer;
        int colStride = colGradients.ColumnsStride;
        float* gradPtr = paddedGrad.Pointer;

        var padW = paddedW;
        var padWH = paddedH * paddedW;
        var padWHC = Channels * paddedH * paddedW;

        for (int b = 0; b < Batch; ++b)
        {
            long batchOffsetGrad = b * padWHC;

            for (int oh = 0; oh < Height; oh++)
            {
                long patchRowBase = (b * Height + oh) * Width;

                for (int ow = 0; ow < Width; ow++)
                {
                    float* colRow = colPtr + (patchRowBase + ow) * colStride;

                    for (int c = 0; c < Channels; c++)
                    {
                        long channelOffsetGrad = (c * padWH) + batchOffsetGrad;
                        long channelOffsetCol = c * kernelSpatial;

                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            float* dstGrad = gradPtr + channelOffsetGrad + ((oh + ky) * padW) + ow;
                            float* srcCol = colRow + channelOffsetCol + (ky * kernelW);

                            for (int kx = 0; kx < kernelW; ++kx)
                            {
                                dstGrad[kx] += srcCol[kx];
                            }
                        }
                    }
                }
            }
        }

        nuint rowBytes = (nuint)Width * sizeof(float);
        float* baseDstPtr = Pointer;
        float* basePaddedGradPtr = paddedGrad.Pointer;

        for (int b = 0; b < Batch; ++b)
        {
            long batchSrcOffset = b * padWHC;
            long batchDstOffset = (long)b * Channels * Height * Width;

            for (int c = 0; c < Channels; c++)
            {
                float* srcChannel = basePaddedGradPtr + batchSrcOffset + (c * padWH);
                float* dstChannel = baseDstPtr + batchDstOffset + (c * Height * Width);

                for (int y = 0; y < Height; y++)
                {
                    float* srcPtr = srcChannel + ((y + 1) * padW) + 1;
                    float* dstPtr = dstChannel + (y * Width);
                    NativeMemory.Copy(srcPtr, dstPtr, rowBytes);
                }
            }
        }
    }

    private bool _isDisposing = false;
    [OverloadResolutionPriority(1)]
    public void Dispose([CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        try
        {
            if (_isDisposing || _isInit)
            {
                throw new InvalidOperationException("ooga");
            }

            _isDisposing = true;
            EnsureNotDisposed();

            DisposeLocations?.Add(SourceLocation.Current(new MatrixInfo([Batch, Channels, Height, Width], UnsafeSize), fp, ln));
            _inUse = false;

            MemoryHandle.Take().Dispose();
            GC.SuppressFinalize(this);

            // if (_isPoolable)
            // {
            //     _pool.Add(this);
            // }
            // else
            // {
            //     NativeMemory.AlignedFree(Pointer);
            //     GC.SuppressFinalize(this);
            //     Pointer = null;
            // }

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
        Console.WriteLine(Locations?[^1]);
        MemoryHandle.Free();
    }
}
