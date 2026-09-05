using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Convolutional;

/// <summary>
/// Superoptimized 4D tensor (Batch, Channels, Height, Width) with zero-alloc L1-cached memory pooling.
/// </summary>
public unsafe class CnnMatrix : IDisposable
{
    public const int Alignment = 16;
    private const int ByteAlignment = Alignment * sizeof(float);

    private static readonly ConcurrentBag<CnnMatrix> _pool = [];
    private static readonly int CommonAllocatedLength = 0x400000; // 2,621,440 floats

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
        if (!_pool.TryTake(out var item))
        {
            return new CnnMatrix(batch, channels, height, width, readOnly);
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

        Pointer = (float*)NativeMemory.AlignedAlloc((nuint)(CommonAllocatedLength * sizeof(float)), (nuint)ByteAlignment);
        _inUse = true;
        Clear();
    }

    public void Resize(int batch, int channels, int height, int width)
    {
        int newSize = batch * channels * height * width;
        if (newSize > CommonAllocatedLength)
        {
            throw new InvalidOperationException($"Tensor size {newSize} exceeds pool buffer size {CommonAllocatedLength}.");
        }

        Batch = batch;
        Channels = channels;
        Height = height;
        Width = width;
        UnsafeSize = newSize;
        _inUse = true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetIndex(int batch, int channel, int y, int x)
        => (batch * StrideN) + (channel * StrideC) + (y * StrideH) + x;

    public ref float this[int batch, int channel, int y, int x]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref Pointer[GetIndex(batch, channel, y, x)];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetChannelPointer(int batch, int channel) => Pointer + (batch * StrideN) + (channel * StrideC);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetRowPointer(int batch, int channel, int y) => Pointer + (batch * StrideN) + (channel * StrideC) + (y * StrideH);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Clear()
    {
        NativeMemory.Clear(Pointer, (nuint)UnsafeSize * sizeof(float));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Fill(float value)
    {
        new Span<float>(Pointer, UnsafeSize).Fill(value);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void CopyFrom(CnnMatrix other)
    {
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

        var colMatrix = NeuralMatrix.GetOrCreate(totalPatches, patchSize);
        float* colPtr = colMatrix.Pointer;
        int colStride = colMatrix.ColumnsStride;

        using var padded = GetOrCreate(Batch, Channels, paddedH, paddedW);

        // Fast zero-fill only if padding is required
        if (padding > 0)
        {
            padded.Clear();
        }

        nuint rowBytes = (nuint)Width * sizeof(float);

        // Optimized Cache-Line Sequential Copy for Input Padding
        for (int b = 0; b < Batch; b++)
        {
            for (int c = 0; c < Channels; c++)
            {
                float* srcPtr = GetChannelPointer(b, c);
                float* dstPtr = padded.GetChannelPointer(b, c);

                for (int y = 0; y < Height; y++)
                {
                    float* srcRow = srcPtr + y * Width;
                    float* dstRow = dstPtr + (y + padding) * paddedW + padding;
                    NativeMemory.Copy(srcRow, dstRow, rowBytes);
                }
            }
        }

        // Parallelized SIMD Block Extraction straight into Column Matrix Memory
        for (int b = 0; b < Batch; b++)
        {
            int batchPatchBase = b * outH * outW;

            for (int oh = 0; oh < outH; oh++)
            {
                int startY = oh * stride;
                int patchRowBase = batchPatchBase + oh * outW;

                for (int ow = 0; ow < outW; ow++)
                {
                    int startX = ow * stride;
                    float* dstRow = colPtr + (patchRowBase + ow) * colStride;
                    int colIdx = 0;

                    for (int c = 0; c < Channels; c++)
                    {
                        float* paddedPtr = padded.GetChannelPointer(b, c);

                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            float* srcRow = paddedPtr + (startY + ky) * paddedW + startX;

                            if (kernelW == 3 && Avx2.IsSupported)
                            {
                                // Vectorized fast-path for standard 3x3 kernels
                                Vector128<float> kVec = Sse.LoadLow(Vector128<float>.Zero, srcRow);
                                kVec = Sse.LoadHigh(kVec, srcRow + 1); // load 3 floats
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

        int kernelSpatial = kernelH * kernelW;

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
                                    vDst = Avx512F.FusedMultiplyAdd(vSrc, vScale512, vDst);
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
                                    vDst = Fma.IsSupported
                                        ? Fma.MultiplyAdd(vSrc, vScale256, vDst)
                                        : Avx.Add(vDst, Avx.Multiply(vSrc, vScale256));
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
        }

        nuint rowBytes = (nuint)Width * sizeof(float);
        for (int b = 0; b < Batch; b++)
        {
            for (int c = 0; c < Channels; c++)
            {
                for (int y = 0; y < Height; y++)
                {
                    float* srcPtr = paddedGrad.GetRowPointer(b, c, y + padding) + padding;
                    float* dstPtr = GetRowPointer(b, c, y);
                    NativeMemory.Copy(srcPtr, dstPtr, rowBytes);
                }
            }
        }
    }

    public void Dispose()
    {
        _inUse = false;
        _pool.Add(this);
    }

    public static void ClearPool()
    {
        while (_pool.TryTake(out var item))
        {
            if (item.Pointer != null)
            {
                NativeMemory.AlignedFree(item.Pointer);
                item.Pointer = null;
            }
        }
    }
}
