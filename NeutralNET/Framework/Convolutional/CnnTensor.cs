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
    public List<SourceLocation> Locations = [];
    public List<SourceLocation> DisposeLocations = [];

    private string DebugLocations => string.Join('\n', Locations
        .Select(x => ("locations", x))
        .Concat(DisposeLocations.Select(x => ("dispose_locations", x)))
        .OrderBy(x => x.x.TimeStamp));

    public const int Alignment = 16;
    private const int ByteAlignment = Alignment * sizeof(float);

    // private static readonly ConcurrentBag<CnnMatrix> _pool = [];
    // private static readonly int CommonAllocatedLength = 134_217_728;

    public static readonly ConcurrentBag<CnnMatrix> Instances = [];
    public AllocationHandle MemoryHandle;
    public float* Pointer { [MethodImpl(Inline)] get => (float*)MemoryHandle.Pointer; }
    public int Batch { get; set; }
    public int Channels;
    public int Height;
    public int Width;
    public int UnsafeSize;
    public bool ReadOnly;

    public int StrideW => 1;
    public int StrideH => Width;
    public int StrideC => Width * Height;
    public int StrideN => Width * Height * Channels;

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

        Locations.Add(SourceLocation.Current(new MatrixInfo([batch, channels, height, width], UnsafeSize), fp, ln));
        // Pointer = (float*)NativeMemory.AlignedAlloc((nuint)(allocatedLength * sizeof(float)), (nuint)ByteAlignment);
        _inUse = true;
        Clear();
        Instances.Add(this);
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

        if (colGradients == null) throw new ArgumentNullException(nameof(colGradients));

        // ---- geometry (all long) ----
        long paddedH = Height + 2L * padding;
        long paddedW = Width + 2L * padding;
        long outH = (paddedH - kernelH) / stride + 1;
        long outW = (paddedW - kernelW) / stride + 1;

        if (outH <= 0 || outW <= 0)
            throw new InvalidOperationException($"Col2Im: bad out dims outH={outH} outW={outW}");

        long expectedColRows = (long)Batch * outH * outW;
        long expectedColCols = (long)Channels * kernelH * kernelW;

        if (colGradients.Rows < expectedColRows)
            throw new InvalidOperationException(
                $"Col2Im: colGradients.Rows={colGradients.Rows} < required {expectedColRows}");
        if (colGradients.ColumnsStride < expectedColCols)
            throw new InvalidOperationException(
                $"Col2Im: colGradients.ColumnsStride={colGradients.ColumnsStride} < required {expectedColCols}");

        long colTotalFloats = (long)colGradients.Rows * colGradients.ColumnsStride;
        long dstTotalFloats = (long)Batch * Channels * Height * Width;
        long padTotalFloats = (long)Batch * Channels * paddedH * paddedW;

        long padStrideN = paddedH * paddedW * Channels;
        long padStrideC = paddedH * paddedW;
        long padStrideH = paddedW;
        int colStride = colGradients.ColumnsStride;
        long kernelSpatial = (long)kernelH * kernelW;

        // ---- allocate scratch ----
        var paddedGrad = GetOrCreate(Batch, Channels, (int)paddedH, (int)paddedW);

        // Sanity: the pool must have given us at least as many bytes as we asked for.
        if (paddedGrad.UnsafeSize * sizeof(float) > (int)paddedGrad.MemoryHandle.ByteSize)
            throw new InvalidOperationException(
                $"Col2Im: paddedGrad under-allocated. need={paddedGrad.UnsafeSize * sizeof(float)} bytes, " +
                $"handle={paddedGrad.MemoryHandle.ByteSize} bytes");

        try
        {
            // Clear via the exact float count — never rely on ByteSize, which is in bytes.
            NativeMemory.Clear(paddedGrad.Pointer, (nuint)paddedGrad.UnsafeSize * sizeof(float));

            float* colPtr = colGradients.Pointer;
            float* gradPtr = paddedGrad.Pointer;

            // ============================================================
            // Scatter loop
            // ============================================================
            Parallel.For(0, Batch, b =>
            {
                long batchOffsetGrad = (long)b * padStrideN;

                for (long oh = 0; oh < outH; oh++)
                {
                    long startY = oh * stride;

                    for (long ow = 0; ow < outW; ow++)
                    {
                        long startX = ow * stride;

                        long patchIdx = ((long)b * outH + oh) * outW + ow;
                        long colRowOff = patchIdx * colStride;

                        if (colRowOff < 0 || colRowOff + expectedColCols > colTotalFloats)
                            throw new InvalidOperationException(
                                $"Col2Im colRow OOB b={b} oh={oh} ow={ow} " +
                                $"off={colRowOff} total={colTotalFloats}");

                        float* colRow = colPtr + colRowOff;

                        for (long c = 0; c < Channels; c++)
                        {
                            long channelOffsetGrad = batchOffsetGrad + c * padStrideC;
                            long channelOffsetCol = c * kernelSpatial;

                            for (long ky = 0; ky < kernelH; ky++)
                            {
                                long dstOff = channelOffsetGrad + (startY + ky) * padStrideH + startX;
                                if (dstOff < 0 || dstOff + kernelW > padTotalFloats)
                                    throw new InvalidOperationException(
                                        $"Col2Im dst OOB b={b} c={c} oh={oh} ow={ow} ky={ky} " +
                                        $"off={dstOff} total={padTotalFloats}");

                                long srcOff = colRowOff + channelOffsetCol + ky * kernelW;
                                if (srcOff < 0 || srcOff + kernelW > colTotalFloats)
                                    throw new InvalidOperationException(
                                        $"Col2Im src OOB b={b} c={c} oh={oh} ow={ow} ky={ky} " +
                                        $"off={srcOff} total={colTotalFloats}");

                                float* dstGrad = gradPtr + dstOff;
                                float* srcCol = colPtr + srcOff;

                                long kx = 0;

                                // AVX512 (unaligned-safe)
                                if (Avx512F.IsSupported)
                                {
                                    var vScale = Vector512.Create(scale);
                                    long vecLimit = kernelW - (kernelW % 16);
                                    for (; kx < vecLimit; kx += 16)
                                    {
                                        var vDst = Vector512.Load(dstGrad + kx);
                                        var vSrc = Vector512.Load(srcCol + kx);
                                        vDst = Vector512.FusedMultiplyAdd(vSrc, vScale, vDst);
                                        Vector512.Store(vDst, dstGrad + kx);
                                    }
                                }
                                else if (Avx2.IsSupported)
                                {
                                    var vScale = Vector256.Create(scale);
                                    long vecLimit = kernelW - (kernelW % 8);
                                    for (; kx < vecLimit; kx += 8)
                                    {
                                        var vDst = Avx.LoadVector256(dstGrad + kx);
                                        var vSrc = Avx.LoadVector256(srcCol + kx);
                                        vDst = Fma.IsSupported
                                            ? Fma.MultiplyAdd(vSrc, vScale, vDst)
                                            : Avx.Add(vDst, Avx.Multiply(vSrc, vScale));
                                        Avx.Store(dstGrad + kx, vDst);
                                    }
                                }

                                for (; kx < kernelW; kx++)
                                    dstGrad[kx] += srcCol[kx] * scale;
                            }
                        }
                    }
                }
            });

            // ============================================================
            // Crop-back pass: copy the padded interior into `this`
            // ============================================================
            long rowFloats = Width;
            nuint rowBytes = (nuint)(rowFloats * sizeof(float));

            float* baseDst = Pointer;
            float* baseSrc = paddedGrad.Pointer;

            Parallel.For(0, Batch, b =>
            {
                long batchSrcOff = (long)b * padStrideN;
                long batchDstOff = (long)b * Channels * Height * Width;

                for (long c = 0; c < Channels; c++)
                {
                    long srcBase = batchSrcOff + c * padStrideC;
                    long dstBase = batchDstOff + c * Height * Width;

                    for (long y = 0; y < Height; y++)
                    {
                        long srcOff = srcBase + (y + padding) * padStrideH + padding;
                        long dstOff = dstBase + y * Width;

                        if (srcOff < 0 || srcOff + rowFloats > padTotalFloats)
                            throw new InvalidOperationException(
                                $"Col2Im copyback src OOB b={b} c={c} y={y} off={srcOff}");
                        if (dstOff < 0 || dstOff + rowFloats > dstTotalFloats)
                            throw new InvalidOperationException(
                                $"Col2Im copyback dst OOB b={b} c={c} y={y} off={dstOff}");

                        NativeMemory.Copy(baseSrc + srcOff, baseDst + dstOff, rowBytes);
                    }
                }
            });
        }
        finally
        {
            paddedGrad.Dispose();
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

            DisposeLocations.Add(SourceLocation.Current(new MatrixInfo([Batch, Channels, Height, Width], UnsafeSize), fp, ln));
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
        Console.WriteLine(Locations[^1]);
        MemoryHandle.Free();
    }
}
