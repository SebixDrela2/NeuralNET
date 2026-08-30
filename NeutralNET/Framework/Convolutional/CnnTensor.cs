using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Convolutional;

/// <summary>
/// 4D tensor (Batch, Channels, Height, Width) with native aligned memory and pooling.
/// </summary>
public unsafe class CnnMatrix : IDisposable
{
    public const int Alignment = 16;
    private const int AlignmentMask = Alignment - 1;
    private const int ByteAlignment = Alignment * sizeof(float);
    private const int ByteAlignmentMask = ByteAlignment - 1;

    private static readonly Stack<CnnMatrix> _pool = [];

    // Fixed buffer size - match NeuralMatrix
    private const int CommonAllocatedLength = 0x280000; // 2,621,440 floats = ~10MB

    public float* Pointer;
    public int Batch;
    public int Channels;
    public int Height;
    public int Width;
    public bool ReadOnly;
    public int UnsafeSize;

    public int StrideW => 1;
    public int StrideH => Width;
    public int StrideC => Width * Height;
    public int StrideN => Width * Height * Channels;

    private bool _inUse = true;

    public static CnnMatrix GetOrCreate(int batch, int channels, int height, int width, bool readOnly = false)
    {
        if (!_pool.TryPop(out var item))
        {
            item = new CnnMatrix(batch, channels, height, width, readOnly);
            return item;
        }

        item.Resize(batch, channels, height, width);
        return item;
    }

    private CnnMatrix(int batch, int channels, int height, int width, bool readOnly = false)
    {
        Batch = batch;
        Channels = channels;
        Height = height;
        Width = width;
        ReadOnly = readOnly;
        UnsafeSize = batch * channels * height * width;

        // Allocate aligned memory
        nuint byteCount = ((nuint)(CommonAllocatedLength * sizeof(float)) + ByteAlignmentMask) & ~(uint)ByteAlignmentMask;
        Pointer = (float*)NativeMemory.AlignedAlloc(byteCount, ByteAlignment);
        _inUse = true;
        Clear();
    }

    public void Resize(int batch, int channels, int height, int width)
    {
        var newSize = batch * channels * height * width;
        if (newSize > CommonAllocatedLength)
        {
            throw new InvalidOperationException(
                $"Tensor size {newSize} exceeds pool buffer size {CommonAllocatedLength}. " +
                $"Increase CommonAllocatedLength.");
        }

        if (_inUse)
        {
            throw new InvalidOperationException("Cannot resize a matrix that is currently in use.");
        }

        Batch = batch;
        Channels = channels;
        Height = height;
        Width = width;
        UnsafeSize = newSize;
        _inUse = true;
        Clear();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetIndex(int batch, int channel, int y, int x)
        => (batch * StrideN) + (channel * StrideC) + (y * StrideH) + (x * StrideW);

    public ref float this[int batch, int channel, int y, int x]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            int idx = GetIndex(batch, channel, y, x);
            if (idx >= UnsafeSize || idx < 0)
                throw new IndexOutOfRangeException($"Index {idx} >= {UnsafeSize}");
            return ref Pointer[idx];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetChannelPointer(int batch, int channel)
    {
        int offset = (batch * StrideN) + (channel * StrideC);
        int channelSize = Height * Width;
        if (offset + channelSize > UnsafeSize || offset < 0)
            throw new IndexOutOfRangeException($"Channel offset {offset} + size {channelSize} > {UnsafeSize}");
        return Pointer + offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetRowPointer(int batch, int channel, int y)
    {
        int offset = (batch * StrideN) + (channel * StrideC) + (y * StrideH);
        if (offset + Width > UnsafeSize || offset < 0)
            throw new IndexOutOfRangeException($"Row offset {offset} + width {Width} > {UnsafeSize}");
        return Pointer + offset;
    }

    public void Clear()
    {
        NativeMemory.Clear(Pointer, (nuint)UnsafeSize * sizeof(float));
    }

    public void Fill(float value)
    {
        var span = new Span<float>(Pointer, UnsafeSize);
        span.Fill(value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void CopyFrom(CnnMatrix other)
    {
        if (other.UnsafeSize != UnsafeSize)
            throw new ArgumentException($"Size mismatch: {other.UnsafeSize} != {UnsafeSize}");

        NativeMemory.Copy(other.Pointer, Pointer, (nuint)UnsafeSize * sizeof(float));
    }

    public NeuralMatrix Im2Col(int kernelH, int kernelW, int stride, int padding)
    {
        int paddedH = Height + 2 * padding;
        int paddedW = Width + 2 * padding;
        int outH = (paddedH - kernelH) / stride + 1;
        int outW = (paddedW - kernelW) / stride + 1;
        int patchSize = Channels * kernelH * kernelW;

        int totalPatches = Batch * outH * outW;
        int colMatrixSize = totalPatches * patchSize;

        if (colMatrixSize > CommonAllocatedLength)
        {
            throw new InvalidOperationException(
                $"Col matrix size {colMatrixSize} exceeds pool buffer size {CommonAllocatedLength}. " +
                $"Increase CommonAllocatedLength.");
        }

        var colMatrix = NeuralMatrix.GetOrCreate(totalPatches, patchSize);
        using var padded = GetOrCreate(Batch, Channels, paddedH, paddedW);
        padded.Clear();

        float* colPtr = colMatrix.Pointer;
        int colStride = colMatrix.ColumnsStride;

        // Step 1: Copy with padding
        int rowSizeInBytes = Width * sizeof(float);

        for (int b = 0; b < Batch; b++)
        {
            for (int c = 0; c < Channels; c++)
            {
                float* srcPtr = GetChannelPointer(b, c);
                float* dstPtr = padded.GetChannelPointer(b, c);

                for (int y = 0; y < Height; y++)
                {
                    float* srcRow = srcPtr + y * Width;
                    float* dstRow = dstPtr + (y + padding) * padded.Width + padding;

                    Buffer.MemoryCopy(srcRow, dstRow, rowSizeInBytes, rowSizeInBytes);
                }
            }
        }

        for (int b = 0; b < Batch; b++)
        {
            for (int c = 0; c < Channels; c++)
            {
                float* paddedPtr = padded.GetChannelPointer(b, c);
                int channelOffset = c * kernelH * kernelW;

                for (int oh = 0; oh < outH; oh++)
                {
                    int startY = oh * stride;
                    int patchRowBase = (b * outH + oh) * outW;

                    int ow = 0;

                    // --- AVX-512 Vectorized Path (16 floats / 512-bit) ---
                    if (Avx512F.IsSupported && stride == 1)
                    {
                        for (; ow <= outW - 16; ow += 16)
                        {
                            int startX = ow; // stride == 1

                            for (int ky = 0; ky < kernelH; ky++)
                            {
                                float* srcRow = paddedPtr + (startY + ky) * padded.Width + startX;
                                int kyOffset = ky * kernelW;

                                for (int kx = 0; kx < kernelW; kx++)
                                {
                                    // Load 16 continuous spatial inputs into a 512-bit SIMD register
                                    Vector512<float> vecSrc = Avx512F.LoadVector512(srcRow + kx);
                                    int dstOff = channelOffset + kyOffset + kx;

                                    // Store each of the 16 lanes into its corresponding output patch row
                                    // (Unrolled loop allows the JIT compiler to optimize register extractions)
                                    float* dstRow0 = colPtr + (patchRowBase + ow + 0) * colStride;
                                    float* dstRow1 = colPtr + (patchRowBase + ow + 1) * colStride;
                                    float* dstRow2 = colPtr + (patchRowBase + ow + 2) * colStride;
                                    float* dstRow3 = colPtr + (patchRowBase + ow + 3) * colStride;
                                    float* dstRow4 = colPtr + (patchRowBase + ow + 4) * colStride;
                                    float* dstRow5 = colPtr + (patchRowBase + ow + 5) * colStride;
                                    float* dstRow6 = colPtr + (patchRowBase + ow + 6) * colStride;
                                    float* dstRow7 = colPtr + (patchRowBase + ow + 7) * colStride;
                                    float* dstRow8 = colPtr + (patchRowBase + ow + 8) * colStride;
                                    float* dstRow9 = colPtr + (patchRowBase + ow + 9) * colStride;
                                    float* dstRow10 = colPtr + (patchRowBase + ow + 10) * colStride;
                                    float* dstRow11 = colPtr + (patchRowBase + ow + 11) * colStride;
                                    float* dstRow12 = colPtr + (patchRowBase + ow + 12) * colStride;
                                    float* dstRow13 = colPtr + (patchRowBase + ow + 13) * colStride;
                                    float* dstRow14 = colPtr + (patchRowBase + ow + 14) * colStride;
                                    float* dstRow15 = colPtr + (patchRowBase + ow + 15) * colStride;

                                    dstRow0[dstOff] = vecSrc.GetElement(0);
                                    dstRow1[dstOff] = vecSrc.GetElement(1);
                                    dstRow2[dstOff] = vecSrc.GetElement(2);
                                    dstRow3[dstOff] = vecSrc.GetElement(3);
                                    dstRow4[dstOff] = vecSrc.GetElement(4);
                                    dstRow5[dstOff] = vecSrc.GetElement(5);
                                    dstRow6[dstOff] = vecSrc.GetElement(6);
                                    dstRow7[dstOff] = vecSrc.GetElement(7);
                                    dstRow8[dstOff] = vecSrc.GetElement(8);
                                    dstRow9[dstOff] = vecSrc.GetElement(9);
                                    dstRow10[dstOff] = vecSrc.GetElement(10);
                                    dstRow11[dstOff] = vecSrc.GetElement(11);
                                    dstRow12[dstOff] = vecSrc.GetElement(12);
                                    dstRow13[dstOff] = vecSrc.GetElement(13);
                                    dstRow14[dstOff] = vecSrc.GetElement(14);
                                    dstRow15[dstOff] = vecSrc.GetElement(15);
                                }
                            }
                        }
                    }

                    // --- Scalar Tail Loop (Handles outW remainder or stride > 1) ---
                    for (; ow < outW; ow++)
                    {
                        int startX = ow * stride;
                        int patchRow = patchRowBase + ow;
                        float* dstRow = colPtr + patchRow * colStride;

                        int kyOffset = 0;
                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            float* srcRow = paddedPtr + (startY + ky) * padded.Width + startX;
                            int dstOff = channelOffset + kyOffset;

                            for (int kx = 0; kx < kernelW; kx++)
                                dstRow[dstOff + kx] = srcRow[kx];

                            kyOffset += kernelW;
                        }
                    }
                }
            }
        }

        return colMatrix;
    }

    public void Col2Im(NeuralMatrix colGradients, int kernelH, int kernelW, int stride, int padding, float scale = 1.0f)
    {
        int paddedH = Height + 2 * padding;
        int paddedW = Width + 2 * padding;
        int outH = (paddedH - kernelH) / stride + 1;
        int outW = (paddedW - kernelW) / stride + 1;

        using var paddedGrad = GetOrCreate(Batch, Channels, paddedH, paddedW);
        paddedGrad.Clear();

        float* colPtr = colGradients.Pointer;
        int colStride = colGradients.ColumnsStride;
        float* gradPtr = paddedGrad.Pointer;

        var vScale512 = Vector512.Create(scale);

        // --- 1. Accumulation Loop (col2im scatter-add) ---
        for (int b = 0; b < Batch; b++)
        {
            long batchOffsetGrad = b * paddedGrad.StrideN;

            for (int oh = 0; oh < outH; oh++)
            {
                int startY = oh * stride;
                int patchRowBase = (b * outH + oh) * outW;

                for (int ow = 0; ow < outW; ow++)
                {
                    int startX = ow * stride;
                    int patchRow = patchRowBase + ow;
                    float* colRow = colPtr + patchRow * colStride;

                    for (int c = 0; c < Channels; c++)
                    {
                        long channelOffsetGrad = batchOffsetGrad + c * paddedGrad.StrideC;
                        int channelOffsetCol = c * kernelH * kernelW;

                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            long rowOffsetGrad = channelOffsetGrad + (startY + ky) * paddedGrad.StrideH + startX;
                            int kyOffsetCol = channelOffsetCol + ky * kernelW;

                            float* dstGrad = gradPtr + rowOffsetGrad;
                            float* srcCol = colRow + kyOffsetCol;

                            int kx = 0;

                            // AVX-512 Vectorized accumulation (16 floats at a time)
                            if (Avx512F.IsSupported)
                            {
                                for (; kx <= kernelW - 16; kx += 16)
                                {
                                    var vCol = Avx512F.LoadVector512(srcCol + kx);
                                    var vGrad = Avx512F.LoadVector512(dstGrad + kx);

                                    // vGrad = vGrad + (vCol * vScale)
                                    var vRes = Avx512F.FusedMultiplyAdd(vCol, vScale512, vGrad);
                                    vRes.Store(dstGrad + kx);
                                }
                            }

                            // Scalar Tail Loop
                            for (; kx < kernelW; kx++)
                            {
                                dstGrad[kx] += srcCol[kx] * scale;
                            }
                        }
                    }
                }
            }
        }

        // --- 2. Copy back from padded buffer (Vectorized Unpadded Copy) ---
        for (int b = 0; b < Batch; b++)
        {
            for (int c = 0; c < Channels; c++)
            {
                for (int y = 0; y < Height; y++)
                {
                    float* srcPtr = paddedGrad.GetRowPointer(b, c, y + padding) + padding;
                    float* dstPtr = GetRowPointer(b, c, y);

                    int x = 0;

                    if (Avx512F.IsSupported)
                    {
                        for (; x <= Width - 16; x += 16)
                        {
                            var vec = Avx512F.LoadVector512(srcPtr + x);
                            vec.Store(dstPtr + x);
                        }
                    }

                    // Tail Loop
                    for (; x < Width; x++)
                    {
                        dstPtr[x] = srcPtr[x];
                    }
                }
            }
        }
    }

    public void Dispose()
    {
        _inUse = false;
        _pool.Push(this);
    }

    public static void ClearPool()
    {
        while (_pool.TryPop(out var item))
        {
            if (item.Pointer != null)
            {
                NativeMemory.AlignedFree(item.Pointer);
                item.Pointer = null;
            }
        }
    }
}
