using System.Diagnostics;
using System.Numerics;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Stuff;
using NeutralNET.Unmanaged;
using NeutralNET.Utils;

namespace NeutralNET.Matrices;

/// <summary>
/// High‑performance matrix with SIMD AVX-512/AVX2 acceleration and pooled unsafe buffers.
/// </summary>
public unsafe class NeuralMatrix : CriticalFinalizerObject, IDisposable
{
    public const int Alignment = SIMD.AlignSize;
    private const int ByteAlignment = SIMD.ByteAlignSize;

    public AllocationHandle MemoryHandle;
    public int Rows;
    public int ColumnsStride;
    public int UsedColumns;
    public int LogicalLength;
    public uint[] StrideMasks;
    public int UnsafeSize;

    private bool _inUse = true;
    private readonly bool _isPoolable = true;

    public string? DisplayName { get; set; }

    public float* Pointer { [MethodImpl(Inline)] get => (float*)MemoryHandle.Pointer; }
    public Span<float> this[int row] { [MethodImpl(Inline)] get => new(Pointer + (row * ColumnsStride), UsedColumns); }
    public ref float this[int row, int col] { [MethodImpl(Inline)] get => ref Pointer[(row * ColumnsStride) + col]; }

    public Span<float> SpanWithGarbage => new(Pointer, UnsafeSize);

    public static NeuralMatrix GetOrCreate(int rows, int columns, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
        => new(rows, columns, fp, ln);

    private NeuralMatrix(int rows, int columns, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        ColumnsStride = MatrixUtils.GetStride(columns);
        Rows = rows;
        UsedColumns = columns;

        LogicalLength = Rows * UsedColumns;
        UnsafeSize = Rows * ColumnsStride;

        MemoryHandle = NeuralMemoryPool.Rent<float>(UnsafeSize);
        StrideMasks = MatrixUtils.GetStrideMask(columns);

        Clear();
    }

    public void SetRowSize(int limit)
    {
        Rows = limit;
        LogicalLength = Rows * UsedColumns;
        UnsafeSize = Rows * ColumnsStride;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void DotVectorized(NeuralMatrix other, NeuralMatrix result)
    {
        // Standard matmul: A[m, k] @ B[k, n] = C[m, n]
        int m = Rows;
        int k = UsedColumns;
        int n = other.UsedColumns;

        float* pA = Pointer; int aStride = ColumnsStride;
        float* pB = other.Pointer; int bStride = other.ColumnsStride;
        float* pR = result.Pointer; int rStride = result.ColumnsStride;

        Parallel.For(0, m, i =>
        {
            float* aRow = pA + i * aStride;
            float* rRow = pR + i * rStride;

            // Zero the output row (only the used columns; padding can stay whatever).
            for (int j = 0; j < n; j++) rRow[j] = 0f;

            // C[i, :] += A[i, p] * B[p, :] for each p in [0, k)
            // This is a row-of-A scalars times rows-of-B, accumulated into C's row.
            for (int p = 0; p < k; p++)
            {
                float aVal = aRow[p];
                if (aVal == 0f) continue;

                float* bRow = pB + p * bStride;
                int j = 0;

                if (Avx512F.IsSupported)
                {
                    var vA = Vector512.Create(aVal);
                    int vecLimit = n - (n % 16);
                    for (; j < vecLimit; j += 16)
                    {
                        var rVec = Vector512.Load(rRow + j);
                        var bVec = Vector512.Load(bRow + j);
                        rVec = Avx512F.FusedMultiplyAdd(vA, bVec, rVec);
                        rVec.Store(rRow + j);
                    }
                }
                else if (Avx2.IsSupported)
                {
                    var vA = Vector256.Create(aVal);
                    int vecLimit = n - (n % 8);
                    for (; j < vecLimit; j += 8)
                    {
                        var rVec = Vector256.Load(rRow + j);
                        var bVec = Vector256.Load(bRow + j);
                        rVec = Fma.IsSupported
                            ? Fma.MultiplyAdd(vA, bVec, rVec)
                            : Avx.Add(rVec, Avx.Multiply(vA, bVec));
                        rVec.Store(rRow + j);
                    }
                }

                for (; j < n; j++)
                {
                    rRow[j] += aVal * bRow[j];
                }
            }
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void DotTranspose(NeuralMatrix other, NeuralMatrix result)
    {
        // Dla A * B^T: Liczba kolumn A musi zgadzać się z liczbą kolumn B
        if (UsedColumns != other.UsedColumns)
        {
            throw new ArgumentException($"Dimension mismatch for DotTranspose: Left columns ({UsedColumns}) != Right columns ({other.UsedColumns})");
        }

        int batchSize = Rows;
        int outFeatures = other.Rows;
        int inFeatures = UsedColumns;

        float* pInput = Pointer;
        float* pResult = result.Pointer;
        float* pOther = other.Pointer;

        int inStride = ColumnsStride;
        int resStride = result.ColumnsStride;
        int othStride = other.ColumnsStride;

        Parallel.For(0, batchSize, row =>
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
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> GetRowSpan(int row) => SpanWithGarbage.Slice(row * ColumnsStride, UsedColumns);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public NeuralVector GetMatrixRow(int row) => new(GetRowPointer(row), UsedColumns, ColumnsStride);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetRowPointer(int row) => Pointer + (row * ColumnsStride);

    public void CopyRowFrom(NeuralMatrix other, int row) => other.GetRowSpan(row).CopyTo(GetRowSpan(row));

    [Conditional("DEBUG")]
    public static void AssertSameSize(NeuralMatrix lhs, NeuralMatrix rhs, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        AllocationHandle.AssertSameSize(lhs.MemoryHandle, rhs.MemoryHandle, fp, ln);
    }
    public void CopyFrom(NeuralMatrix other, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        AssertSameSize(this, other, fp, ln);
        NativeMemory.Copy(other.Pointer, Pointer, nuint.Min(MemoryHandle.ByteSize, other.MemoryHandle.ByteSize));
    }

    public NeuralMatrix Copy([CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        var matrix = GetOrCreate(Rows, UsedColumns, fp, ln);
        matrix.CopyFrom(this, fp, ln);
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
        NativeMemory.Clear(Pointer, MemoryHandle.ByteSize);
    }

    public override string ToString() => $"{Rows}x{UsedColumns}";

    private const int GpuExecutionThresholdElements = 65536;

    public void Dot(NeuralMatrix other, NeuralMatrix result)
    {
        if (UsedColumns != other.Rows)
        {
            throw new ArgumentException($"Dimension mismatch: Left columns ({UsedColumns}) != Right rows ({other.Rows})");
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

    public void Dispose()
    {
        ObjectDisposedException.ThrowIf(!_inUse, this);

        MemoryHandle.Take().Dispose();
        GC.SuppressFinalize(this);
        _inUse = false;
    }

    ~NeuralMatrix()
    {
        MemoryHandle.Free();
    }
}
