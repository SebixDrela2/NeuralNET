using NeutralNET.Stuff;
using NeutralNET.Unmanaged;
using NeutralNET.Utils;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Matrices;

public unsafe readonly struct NeuralMatrix
{
    public const int Alignment = 8;

    private const int AlignmentMask = Alignment - 1;
    private const int ByteAlignment = Alignment * sizeof(float);
    private const int ByteAlignmentMask = ByteAlignment - 1;

    public static int AllocCounter = 0;
    public static Dictionary<string, StackTrace> Traces = [];
    public static HashSet<(int Width, int Height)> Sizes = [];

    public readonly float* Pointer;

    public readonly int Rows;

    public readonly bool HasStride => ColumnsStride != UsedColumns;

    public readonly int ColumnsStride;
    public readonly int UsedColumns;
    public readonly int LogicalLength;
    public readonly int AllocatedLength;
    public readonly uint[] StrideMasks;
    public Span<float> SpanWithGarbage => new(Pointer, AllocatedLength);

    public NeuralMatrix(int rows, int columns)
    {
        ColumnsStride = MatrixUtils.GetStride(columns);

        Rows = rows;
        UsedColumns = columns;

        LogicalLength = Rows * UsedColumns;
        AllocatedLength = Rows * ColumnsStride;

        nuint byteCount = ((nuint)(AllocatedLength * sizeof(float)) + ByteAlignmentMask) & (~(uint)ByteAlignmentMask);

        Pointer = (float*)NativeMemory.AlignedAlloc(byteCount, ByteAlignment);

        StrideMasks = MatrixUtils.GetStrideMask(columns);
        SpanWithGarbage.Clear();
    }

    public float[] ToArray()
    {
        Debug.Assert(!HasStride);

        return SpanWithGarbage.ToArray();
    }

    public void Dispose() => NativeMemory.AlignedFree(Pointer);

    [Conditional("DEBUG")]
    public static void LogOrigin(int rows, int columns)
    {
        var i = AllocCounter++;

        switch (i)
        {
            case > 1_000_000 when (i % 1_000_000) is not 0:
            case > 100_000 when (i % 100_000) is not 0:
            case > 10_000 when (i % 10_000) is not 0:
            case > 1_000 when (i % 1_000) is not 0:
            case > 100 when (i % 100) is not 0:
            case > 10 when (i % 10) is not 0:
                break;
            default:
                Console.WriteLine($"NEW ARRAY CREATED {i}");
                break;
        }

        var stack = new StackTrace();
        var frames = string.Join("\n", stack.GetFrames().Reverse()
            .Select(x => $"{x.GetMethod()?.DeclaringType?.FullName}.{x.GetMethod()?.Name}"));
        Traces.TryAdd(frames, stack);

        Sizes.Add((columns, rows));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void DotVectorized(NeuralMatrix other, NeuralMatrix result)
    {
        int inFeatures = UsedColumns;
        int outFeatures = other.Rows;
        int batchSize = Rows;
        int vecSize = NeuralMatrix.Alignment;

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
        var result = new NeuralMatrix(Rows, other.UsedColumns);

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
        NativeMemory.Copy(other.Pointer, Pointer, (nuint)AllocatedLength * sizeof(float));
    }

    public void SumVectorized(NeuralMatrix other)
    {
        Debug.Assert(Rows == other.Rows);
        Debug.Assert(UsedColumns == other.UsedColumns);

        var zipPointer = new Zip2Pointer(Pointer, other.Pointer, AllocatedLength);

        if (Avx2.IsSupported)
        {
            while (zipPointer.IsInScope)
            {
                zipPointer.GetVectors(out var aVec, out var bVec);

                var resultVec = Avx512F.Add(aVec, bVec);
                resultVec.StoreAligned(zipPointer.A);

                zipPointer += NeuralMatrix.Alignment;
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
        float* end = ptr + AllocatedLength;

        if (Avx2.IsSupported)
        {
            var meanVec = Vector512.Create(mean);
            var stddevVec = Vector512.Create(stddev);
            var multiplierVec = Vector512.Create(multiplier);

            while (ptr + NeuralMatrix.Alignment <= end)
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

                float[] u1Array = new float[8];
                fixed (float* u1Ptr = u1Array)
                {
                    Avx512F.Store(u1Ptr, u1);
                    for (int i = 0; i < 8; i++)
                        u1Ptr[i] = MathF.Log(u1Ptr[i]);
                }

                // DANIEL PLZ FIX
                var logU1Vec = Vector512.Create(
                    u1Array[0], u1Array[1], u1Array[2], u1Array[3],
                    u1Array[4], u1Array[5], u1Array[6], u1Array[7],
                    u1Array[0], u1Array[1], u1Array[2], u1Array[3],
                    u1Array[4], u1Array[5], u1Array[6], u1Array[7]);

                var sqrtPart = Avx512F.Sqrt(Avx512F.Multiply(Vector512.Create(-2.0f), logU1Vec));

                float[] sinInput = new float[8];
                float[] sinOutput = new float[8];
                fixed (float* sinInputPtr = sinInput)
                {
                    Avx512F.Multiply(Vector512.Create(2.0f * MathF.PI), u2).Store(sinInputPtr);
                    for (int i = 0; i < 8; i++)
                        sinInput[i] = MathF.Sin(sinInput[i]);
                }

                var sinVec = Vector512.Create(
                    sinInput[0], sinInput[1], sinInput[2], sinInput[3],
                    sinInput[4], sinInput[5], sinInput[6], sinInput[7]);

                var z0 = Avx512F.Multiply(sqrtPart, sinVec);

                z0 = Avx512F.FusedMultiplyAdd(
                    Avx512F.Multiply(z0, stddevVec),
                    multiplierVec,
                    meanVec);

                Avx512F.StoreAligned(ptr, z0);
                ptr += NeuralMatrix.Alignment;
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
    public void Clip(float maxNorm)
    {
        Clip(-maxNorm, maxNorm);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ClipVectorized(float min, float max)
    {
        float* ptr = Pointer;
        float* end = ptr + AllocatedLength;

        var minVec = Vector512.Create(min);
        var maxVec = Vector512.Create(max);

        if (Avx2.IsSupported)
        {
            for (; ptr != end; ptr += NeuralMatrix.Alignment)
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
        NativeMemory.Clear(Pointer, (nuint)AllocatedLength * sizeof(float));
    }

    public void Fill(float value)
    {
        float* ptr = Pointer;
        float* end = ptr + AllocatedLength;

        var vec = Vector512.Create(value);

        for (; ptr != end; ptr += NeuralMatrix.Alignment)
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