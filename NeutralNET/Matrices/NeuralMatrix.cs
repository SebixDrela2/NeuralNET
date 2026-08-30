using System.Collections.Concurrent;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Stuff;
using NeutralNET.Unmanaged;
using NeutralNET.Utils;

namespace NeutralNET.Matrices;

/// <summary>
/// High‑performance matrix with AVX‑512 support and a buffer pool for reuse.
/// </summary>
public unsafe class NeuralMatrix : CriticalFinalizerObject, IDisposable
{
    ~NeuralMatrix()
    {
        Console.WriteLine($"JEST ZLE: {Rows}x{UsedColumns}: {CallStack}");
        NativeMemory.AlignedFree(Pointer);
    }

    public const int Alignment = 16;

    private const int AlignmentMask = Alignment - 1;
    private const int ByteAlignment = Alignment * sizeof(float);
    private const int ByteAlignmentMask = ByteAlignment - 1;

    // ---------- Buffer Pool ----------
    private static readonly Stack<NeuralMatrix> _pool = [];

    private static readonly int CommonAllocatedLength = 0x280000;
    private readonly int _allocatedLength;

    public float* Pointer;

    public int Rows;
    public int ColumnsStride;
    public int UsedColumns;
    public int LogicalLength;
    public uint[] StrideMasks;
    public int UnsafeSize;

    public Span<float> SpanWithGarbage => new(Pointer, UnsafeSize);

    public static NeuralMatrix GetOrCreate(int rows, int columns, [CallerLineNumber] int ln = -1)
    {
        if (!_pool.TryPop(out var item))
        {
            item = new NeuralMatrix(rows, columns);
        }
        else
        {
            item.Resize(rows, columns);
        }

        item.CallStack = $"[{ln}]:{new StackTrace()}";

        return item;
    }

    private string CallStack;

    private NeuralMatrix(int rows, int columns)
    {
        ColumnsStride = MatrixUtils.GetStride(columns);
        Rows = rows;
        UsedColumns = columns;

        LogicalLength = Rows * UsedColumns;
        _allocatedLength = CommonAllocatedLength;
        UnsafeSize = Rows * ColumnsStride;

        Pointer = (float*)NativeMemory.AlignedAlloc((nuint)_allocatedLength * sizeof(float), ByteAlignment);
        StrideMasks = MatrixUtils.GetStrideMask(columns);
        SpanWithGarbage.Clear();
    }

    private bool _inUse = true;

    public void Dispose()
    {
        if (!_inUse)
        {
            throw new NotImplementedException("JEst bardoz zle");
        }

        if (_pool.Count > 1024)
        {
            Console.WriteLine("jest zle");
        }

        _inUse = false;
        _pool.Push(this);
    }

    public void Resize(int rows, int columns)
    {
        ColumnsStride = MatrixUtils.GetStride(columns);
        Rows = rows;
        UsedColumns = columns;

        LogicalLength = Rows * UsedColumns;

        if (_allocatedLength < CommonAllocatedLength)
        {
            //Pointer = RentBuffer(AllocatedLength);
            throw new Exception();
        }

        if (_inUse)
        {
            throw new NotImplementedException("JEst bardoz zle grubas");
        }

        _inUse = true;
        UnsafeSize = Rows * ColumnsStride;


        StrideMasks = MatrixUtils.GetStrideMask(columns);
        SpanWithGarbage.Clear();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void DotVectorized(NeuralMatrix other, NeuralMatrix result)
    {
        int inFeatures = UsedColumns;
        int outFeatures = other.Rows;
        int batchSize = Rows;
        int vecSize = Alignment;

        var inputRow = GetMatrixRow(0);
        var resultRow = result.GetMatrixRow(0);

        for (int row = 0; row < batchSize; row++, ++inputRow, ++resultRow)
        {
            var weights = other.GetMatrixRow(0);

            for (int neuronIdx = 0; neuronIdx < outFeatures; neuronIdx++, ++weights)
            {
                var sum = 0f;
                var k = 0;

                var sumVec = Vector512<float>.Zero;
                var vectorizable = inFeatures - (inFeatures % vecSize);

                for (; k < vectorizable; k += vecSize)
                {
                    var inputVec = inputRow.LoadVectorAligned(k);
                    var weightVec = weights.LoadVectorAligned(k);
                    sumVec = Avx512F.FusedMultiplyAdd(inputVec, weightVec, sumVec);
                }

                sum += Vector512.Sum(sumVec);

                for (; k < inFeatures; k++)
                {
                    sum += inputRow.Span[k] * weights.Span[k];
                }

                resultRow.Span[neuronIdx] = sum;
            }
        }
    }

    [Obsolete]
    public NeuralMatrix Dot(NeuralMatrix other)
    {
        if (UsedColumns != other.Rows)
        {
            throw new ArgumentException($"Rows of current: {Rows} do not match other Columns {other.UsedColumns}");
        }
        var innerColumnSize = UsedColumns;
        var result = GetOrCreate(Rows, other.UsedColumns);

        for (var row = 0; row < result.Rows; row++)
        {
            for (var column = 0; column < result.UsedColumns; column++)
            {
                result.Set(row, column, 0);

                for (var k = 0; k < innerColumnSize; k++)
                {
                    var innerAt = At(row, k);
                    var outerAt = other.At(k, column);
                    var multipliedResult = innerAt * outerAt;

                    result.Add(row, column, multipliedResult);
                }
            }
        }

        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> GetRowSpan(int row) => SpanWithGarbage.Slice(row * ColumnsStride, UsedColumns);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public NeuralVector GetMatrixRow(int row)
    {
        float* rowPtr = GetRowPointer(row);
        return new NeuralVector(rowPtr, UsedColumns, ColumnsStride);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetRowPointer(int row) => row * ColumnsStride + Pointer;

    public void CopyRowFrom(NeuralMatrix other, int row)
    {
        other.GetRowSpan(row).CopyTo(GetRowSpan(row));
    }

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

        var zipPointer = new Zip2Pointer(Pointer, other.Pointer, UnsafeSize);

        if (Avx2.IsSupported)
        {
            while (zipPointer.IsInScope)
            {
                zipPointer.GetVectors(out var aVec, out var bVec);

                var resultVec = Avx512F.Add(aVec, bVec);
                resultVec.StoreAligned(zipPointer.A);

                zipPointer += Alignment;
            }
        }
        else
        {
            while (zipPointer.IsInScope)
            {
                *zipPointer.A += *zipPointer.B;
                ++zipPointer;
            }
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

    public void Randomize(float low = 0, float high = 1)
    {
        var span = SpanWithGarbage;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = RandomUtils.GetFloat(1) * (high - low) + low;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RandomizeGaussian(float mean = 0f, float stddev = 1f, float multiplier = 1f, int? seed = null)
    {
        float* ptr = Pointer;
        float* end = ptr + UnsafeSize;

        if (Avx2.IsSupported)
        {
            var meanVec = Vector512.Create(mean);
            var stddevVec = Vector512.Create(stddev);
            var multiplierVec = Vector512.Create(multiplier);

            while (ptr + Alignment <= end)
            {
                var u1 = Vector512.Create(
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed));

                var u2 = Vector512.Create(
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed),
                    RandomUtils.GetFloat(multiplier, seed));

                u1 = Avx512F.Max(u1, Vector512.Create(1e-38f));

                float[] u1Array = new float[Alignment];
                fixed (float* u1Ptr = u1Array)
                {
                    Avx512F.Store(u1Ptr, u1);
                    for (int i = 0; i < Alignment; i++)
                    {
                        u1Ptr[i] = MathF.Log(u1Ptr[i]);
                    }
                }

                // DANIEL PLZ FIX
                var logU1Vec = Vector512.Create(
                    u1Array[0], u1Array[1], u1Array[2], u1Array[3],
                    u1Array[4], u1Array[5], u1Array[6], u1Array[7],
                    u1Array[8], u1Array[9], u1Array[10], u1Array[11],
                    u1Array[12], u1Array[13], u1Array[14], u1Array[15]);

                var sqrtPart = Avx512F.Sqrt(Avx512F.Multiply(Vector512.Create(-2.0f), logU1Vec));

                float[] sinInput = new float[Alignment];
                float[] sinOutput = new float[Alignment];
                fixed (float* sinInputPtr = sinInput)
                {
                    Avx512F.Multiply(Vector512.Create(2.0f * MathF.PI), u2).Store(sinInputPtr);
                    for (int i = 0; i < Alignment; i++)
                    {
                        sinInput[i] = MathF.Sin(sinInput[i]);
                    }
                }

                var sinVec = Vector512.Create(
                    sinInput[0], sinInput[1], sinInput[2], sinInput[3],
                    sinInput[4], sinInput[5], sinInput[6], sinInput[7],
                    sinInput[8], sinInput[9], sinInput[10], sinInput[11],
                    sinInput[12], sinInput[13], sinInput[14], sinInput[15]);

                var z0 = Avx512F.Multiply(sqrtPart, sinVec);

                z0 = Avx512F.FusedMultiplyAdd(
                    Avx512F.Multiply(z0, stddevVec),
                    multiplierVec,
                    meanVec);

                Avx512F.StoreAligned(ptr, z0);
                ptr += Alignment;
            }
        }

        while (ptr < end)
        {
            *ptr++ = RandomUtils.GetGaussian(mean, stddev) * multiplier;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Clip(float min, float max)
    {
        var span = SpanWithGarbage;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = Math.Clamp(span[i], min, max);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Clip(float maxNorm) => Clip(-maxNorm, maxNorm);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ClipVectorized(float min, float max)
    {
        float* ptr = Pointer;
        float* end = ptr + UnsafeSize;

        var minVec = Vector512.Create(min);
        var maxVec = Vector512.Create(max);

        if (Avx2.IsSupported)
        {
            for (; ptr != end; ptr += Alignment)
            {
                var vec = Vector512.LoadAligned(ptr);
                vec = Avx512F.Min(maxVec, Avx512F.Max(minVec, vec));
                vec.StoreAligned(ptr);
            }
        }
        else
        {
            Clip(min, max);
        }
    }

    public void Clear()
    {
        NativeMemory.Clear(Pointer, (nuint)UnsafeSize * sizeof(float));
    }

    public void Fill(float value)
    {
        float* ptr = Pointer;
        float* end = ptr + UnsafeSize;

        var vec = Vector512.Create(value);

        for (; ptr != end; ptr += Alignment)
        {
            vec.StoreAligned(ptr);
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

        Console.WriteLine("]");
        Console.WriteLine();
        Console.WriteLine();
    }

    public override string ToString() => $"{Rows}x{UsedColumns}";
}
