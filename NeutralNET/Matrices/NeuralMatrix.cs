using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.GPU;
using NeutralNET.Stuff;
using NeutralNET.Unmanaged;
using NeutralNET.Utils;

namespace NeutralNET.Matrices;

/// <summary>
/// High‑performance matrix with SIMD AVX-512/AVX2 acceleration and pooled unsafe buffers.
/// </summary>
public unsafe class NeuralMatrix : IDisposable
{
    public const int Alignment = 16;
    private const int ByteAlignment = Alignment * sizeof(float);

    private static readonly ConcurrentBag<NeuralMatrix> _pool = [];
    private static readonly int CommonAllocatedLength = 6422528;

    private readonly int _allocatedLength;

    public float* Pointer;
    public int Rows;
    public int ColumnsStride;
    public int UsedColumns;
    public int LogicalLength;
    public uint[] StrideMasks;
    public int UnsafeSize;

    private bool _inUse = true;

    public Span<float> SpanWithGarbage => new(Pointer, UnsafeSize);

    public static NeuralMatrix GetOrCreate(int rows, int columns)
    {
        if (!_pool.TryTake(out var item))
        {
            item = new NeuralMatrix(rows, columns);
        }
        else
        {
            item.Resize(rows, columns);
        }

        return item;
    }

    private NeuralMatrix(int rows, int columns)
    {
        ColumnsStride = MatrixUtils.GetStride(columns);
        Rows = rows;
        UsedColumns = columns;

        LogicalLength = Rows * UsedColumns;
        _allocatedLength = CommonAllocatedLength;
        UnsafeSize = Rows * ColumnsStride;

        if (UnsafeSize > CommonAllocatedLength)
        {
            throw new InvalidOperationException($"Requested size {UnsafeSize} exceeds CommonAllocatedLength buffer.");
        }

        Pointer = (float*)NativeMemory.AlignedAlloc((nuint)_allocatedLength * sizeof(float), (nuint)ByteAlignment);
        StrideMasks = MatrixUtils.GetStrideMask(columns);
        Clear();
    }

    public void Dispose()
    {
        _inUse = false;
        _pool.Add(this);
    }

    public void Resize(int rows, int columns)
    {
        ColumnsStride = MatrixUtils.GetStride(columns);
        Rows = rows;
        UsedColumns = columns;

        LogicalLength = Rows * UsedColumns;
        UnsafeSize = Rows * ColumnsStride;

        if (UnsafeSize > CommonAllocatedLength)
        {
            throw new InvalidOperationException($"Requested size {UnsafeSize} exceeds CommonAllocatedLength buffer.");
        }

        _inUse = true;
        StrideMasks = MatrixUtils.GetStrideMask(columns);
        Clear();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void DotVectorized(NeuralMatrix other, NeuralMatrix result)
    {
        int inFeatures = UsedColumns;
        int outFeatures = other.Rows;
        int batchSize = Rows;

        float* pInput = Pointer;
        float* pResult = result.Pointer;
        float* pOther = other.Pointer;

        int inStride = ColumnsStride;
        int resStride = result.ColumnsStride;
        int othStride = other.ColumnsStride;

        for (int row = 0; row < batchSize; row++)
        {
            float* inputRow = pInput + row * inStride;
            float* resultRow = pResult + row * resStride;

            for (int neuronIdx = 0; neuronIdx < outFeatures; neuronIdx++)
            {
                float* weights = pOther + neuronIdx * othStride;
                float sum = 0f;
                int k = 0;

                if (Avx512F.IsSupported)
                {
                    var sumVec = Vector512<float>.Zero;
                    int vecLimit = inFeatures - (inFeatures % 16);

                    for (; k < vecLimit; k += 16)
                    {
                        var inputVec = Vector512.Load(inputRow + k);
                        var weightVec = Vector512.Load(weights + k);
                        sumVec = Avx512F.FusedMultiplyAdd(inputVec, weightVec, sumVec);
                    }
                    sum += Vector512.Sum(sumVec);
                }
                else if (Avx2.IsSupported)
                {
                    var sumVec = Vector256<float>.Zero;
                    int vecLimit = inFeatures - (inFeatures % 8);

                    for (; k < vecLimit; k += 8)
                    {
                        var inputVec = Vector256.Load(inputRow + k);
                        var weightVec = Vector256.Load(weights + k);
                        sumVec = Fma.IsSupported
                            ? Fma.MultiplyAdd(inputVec, weightVec, sumVec)
                            : Avx.Add(sumVec, Avx.Multiply(inputVec, weightVec));
                    }

                    var hi = Avx.ExtractVector128(sumVec, 1);
                    var lo = sumVec.GetLower();
                    var sum128 = Sse.Add(lo, hi);
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                    sum += sum128.ToScalar();
                }

                for (; k < inFeatures; k++)
                {
                    sum += inputRow[k] * weights[k];
                }

                resultRow[neuronIdx] = sum;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> GetRowSpan(int row) => SpanWithGarbage.Slice(row * ColumnsStride, UsedColumns);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public NeuralVector GetMatrixRow(int row) => new(GetRowPointer(row), UsedColumns, ColumnsStride);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetRowPointer(int row) => Pointer + (row * ColumnsStride);

    public void CopyRowFrom(NeuralMatrix other, int row) => other.GetRowSpan(row).CopyTo(GetRowSpan(row));

    public void CopyDataFrom(NeuralMatrix other)
    {
        NativeMemory.Copy(other.Pointer, Pointer, (nuint)UnsafeSize * sizeof(float));
    }

    public NeuralMatrix Copy()
    {
        var matrix = GetOrCreate(Rows, UsedColumns);
        matrix.CopyDataFrom(this);
        return matrix;
    }

    public void SumVectorized(NeuralMatrix other)
    {
        Debug.Assert(Rows == other.Rows);
        Debug.Assert(UsedColumns == other.UsedColumns);

        float* pA = Pointer;
        float* pB = other.Pointer;
        int count = UnsafeSize;
        int i = 0;

        if (Avx512F.IsSupported)
        {
            int vecLimit = count - (count % 16);
            for (; i < vecLimit; i += 16)
            {
                var vA = Vector512.Load(pA + i);
                var vB = Vector512.Load(pB + i);
                (vA + vB).Store(pA + i);
            }
        }
        else if (Avx2.IsSupported)
        {
            int vecLimit = count - (count % 8);
            for (; i < vecLimit; i += 8)
            {
                var vA = Vector256.Load(pA + i);
                var vB = Vector256.Load(pB + i);
                (vA + vB).Store(pA + i);
            }
        }

        for (; i < count; i++)
        {
            pA[i] += pB[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref float At(int row, int column) => ref Pointer[row * ColumnsStride + column];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set(int row, int column, float value) => At(row, column) = value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(int row, int column, float value) => At(row, column) += value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Sub(int row, int column, float value) => At(row, column) -= value;

    public void Clear()
    {
        NativeMemory.Clear(Pointer, (nuint)UnsafeSize * sizeof(float));
    }

    public override string ToString() => $"{Rows}x{UsedColumns}";

    private const int GpuExecutionThresholdElements = 65536;

    public void Dot(NeuralMatrix other, NeuralMatrix result)
    {
        if (UsedColumns != other.Rows)
        {
            throw new ArgumentException($"Dimension mismatch: Left columns ({UsedColumns}) != Right rows ({other.Rows})");
        }

        if (UnsafeSize >= GpuExecutionThresholdElements && other.UnsafeSize >= GpuExecutionThresholdElements)
        {
            try
            {
                GpuMatrixOps.ComputeConvolutionGpu(
                    Pointer, other.Pointer, result.Pointer,
                    Rows, other.UsedColumns, UsedColumns,
                    ColumnsStride, other.ColumnsStride, result.ColumnsStride);
                return;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ GPU execution call failed: {ex.Message}. Falling back to CPU SIMD.");
            }
        }

        DotVectorized(other, result);
    }

    public NeuralMatrix Dot(NeuralMatrix other)
    {
        var result = GetOrCreate(Rows, other.UsedColumns);
        Dot(other, result);
        return result;
    }

    public void AddInPlace(NeuralMatrix other) => SumVectorized(other);

    public void Randomize(float low = 0, float high = 1)
    {
        float* ptr = Pointer;
        int count = UnsafeSize;
        float range = high - low;

        for (int i = 0; i < count; i++)
        {
            ptr[i] = RandomUtils.GetFloat(1) * range + low;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RandomizeGaussian(float mean = 0f, float stddev = 1f, float multiplier = 1f, int? seed = null)
    {
        float* ptr = Pointer;
        float* end = ptr + UnsafeSize;

        while (ptr < end)
        {
            *ptr++ = RandomUtils.GetGaussian(mean, stddev) * multiplier;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Clip(float min, float max)
    {
        float* ptr = Pointer;
        int count = UnsafeSize;
        int i = 0;

        if (Avx512F.IsSupported)
        {
            var minVec = Vector512.Create(min);
            var maxVec = Vector512.Create(max);
            int vecLimit = count - (count % 16);

            for (; i < vecLimit; i += 16)
            {
                var vec = Vector512.Load(ptr + i);
                vec = Vector512.Min(maxVec, Vector512.Max(minVec, vec));
                vec.Store(ptr + i);
            }
        }
        else if (Avx2.IsSupported)
        {
            var minVec = Vector256.Create(min);
            var maxVec = Vector256.Create(max);
            int vecLimit = count - (count % 8);

            for (; i < vecLimit; i += 8)
            {
                var vec = Vector256.Load(ptr + i);
                vec = Vector256.Min(maxVec, Vector256.Max(minVec, vec));
                vec.Store(ptr + i);
            }
        }

        for (; i < count; i++)
        {
            ptr[i] = Math.Clamp(ptr[i], min, max);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Clip(float maxNorm) => Clip(-maxNorm, maxNorm);

    public void Fill(float value)
    {
        float* ptr = Pointer;
        int count = UnsafeSize;
        int i = 0;

        if (Avx512F.IsSupported)
        {
            var vec = Vector512.Create(value);
            int vecLimit = count - (count % 16);
            for (; i < vecLimit; i += 16)
            {
                vec.Store(ptr + i);
            }
        }
        else if (Avx2.IsSupported)
        {
            var vec = Vector256.Create(value);
            int vecLimit = count - (count % 8);
            for (; i < vecLimit; i += 8)
            {
                vec.Store(ptr + i);
            }
        }

        for (; i < count; i++)
        {
            ptr[i] = value;
        }
    }

    public void Print(string name)
    {
        Console.WriteLine($"{name} = [");
        for (int i = 0; i < Rows; i++)
        {
            var row = GetRowSpan(i);
            foreach (var val in row)
            {
                Console.Write($"{val,8:F4}");
            }
            Console.WriteLine();
        }
        Console.WriteLine("]\n\n");
    }
}
