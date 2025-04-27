using NeutralNET.Stuff;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;

namespace NeutralNET.Matrices;

public class Matrix
{
    public static Dictionary<string, StackTrace> Traces = [];
    public static HashSet<(int Width, int Height)> Sizes = [];
    
    public int Rows { get; }
    public int Columns { get; }
    public float[] Data { get; set; }
    public Span<float> Span => Data;
    public float FirstElement => Data[0];

    // [Conditional("DEBUG")]
    //private static void LogOrigin(int rows, int columns)
    //{
    //    var i = AllocCounter++;

    //    switch (i)
    //    {
    //        case > 1_000_000 when (i % 1_000_000) is not 0:
    //        case > 100_000 when (i % 100_000) is not 0:
    //        case > 10_000 when (i % 10_000) is not 0:
    //        case > 1_000 when (i % 1_000) is not 0:
    //        case > 100 when (i % 100) is not 0:
    //        case > 10 when (i % 10) is not 0:
    //            break;
    //        default:
    //            Console.WriteLine($"NEW ARRAY CREATED {i}");
    //            break;
    //    }

    //    var stack = new StackTrace();
    //    var frames = string.Join("\n", stack.GetFrames().Reverse()
    //        .Select(x => $"{x.GetMethod()?.DeclaringType?.FullName}.{x.GetMethod()?.Name}"));
    //    Traces.TryAdd(frames, stack);

    //    Sizes.Add((columns, rows));
    //}

    public Matrix(int rows, int columns)
    {
        Rows = rows;
        Columns = columns;
        
        Data = new float[rows * columns];
        //LogOrigin(rows, columns);
    }
    
    public void ApplySigmoid()
    {
        for (int i = 0; i < Data.Length; i++)
        {
            Data[i] = 1f / (1f + float.Exp(-Data[i]));
        }
    }

    public void ApplyReLU()
    {
        for (int i = 0; i < Data.Length; i++)
        {
            Data[i] = float.Max(0.01f * Data[i], Data[i]);
        }
    }

    public void Dot(Matrix other, Matrix result)
    {
        var cols = Columns;
        var otherCols = other.Columns;

        for (int row = 0; row < Rows; ++row)
        {
            for (int col = 0; col < otherCols; ++col)
            {
                float sum = 0f;

                for (int k = 0; k < cols; ++k)
                {
                    sum = float.FusedMultiplyAdd(At(row, k), other.At(k, col), sum);
                }

                result.Set(row, col, sum);
            }
        }       
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Memory<float> GetRowMemory(int row) => Data.AsMemory(row * Columns, Columns);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> GetRowSpan(int row) => Data.AsSpan(row * Columns, Columns);

    public void CopyRowFrom(Matrix other, int row)
    {
        other.GetRowSpan(row).CopyTo(Span);
    }

    public void CopyDataFrom(Matrix other)
    {
        Buffer.BlockCopy(other.Data, 0, Data, 0, Data.Length * sizeof(float));
    }

    public Matrix SplitStart(int column)
    {
        var result = new Matrix(Rows, column);

        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < column; j++)
            {
                result.Set(i, j, At(i, j));
            }
        }

        return result;
    }

    public Matrix SplitEnd(int column)
    {
        var result = new Matrix(Rows, 1);

        for (int i = 0; i < Rows; i++)
        {
            result.Set(i, 0, At(i, column - 1));
        }

        return result;
    }

    public Matrix Row(int row)
    {
        var result = new Matrix(1, Columns);

        Buffer.BlockCopy(Data, row * Columns * sizeof(float),
                       result.Data, 0,
                       Columns * sizeof(float));

        return result;
    }

    public void SumVec(Matrix other)
    {
        nuint length = (nuint)Data.Length;
        nuint vectorSize = (nuint)Vector256<float>.Count;

        nuint i = 0;
        ref var dataReference = ref MemoryMarshal.GetArrayDataReference(Data);
        ref var otherDataReference = ref MemoryMarshal.GetArrayDataReference(other.Data);

        for (; i <= length - vectorSize; i += vectorSize)
        {
            var va = Vector256.LoadUnsafe(ref dataReference, i);
            var vb = Vector256.LoadUnsafe(ref otherDataReference, i);
            var sum = Avx.Add(va, vb);
            sum.StoreUnsafe(ref dataReference, i);
        }

        for (; i < length; ++i)
        {
           Data[i] += other.Data[i];
        }
    }

    public void Sum(Matrix other)
    {
        if (Rows != other.Rows || Columns != other.Columns)
        {
            throw new ArgumentException("Dimension mismatch");
        }

        for (int i = 0; i < Data.Length; i++)
        {
            Data[i] += other.Data[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref float At(int row, int column) => ref Data[row * Columns + column];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set(int row, int column, float value) => Data[row * Columns + column] = value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(int row, int column, float value) => Data[row * Columns + column] += value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Sub(int row, int column, float value) => Data[row * Columns + column] -= value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Divide(int row, int column, float value) => Data[row * Columns + column] /= value;

    public void Randomize(float low = 0, float high = 1)
    {
        for (int i = 0; i < Data.Length; i++)
        {
            Data[i] = RandomUtils.GetFloat(1) * (high - low) + low;
        }
    }

    public void Clear()
    {
        Array.Clear(Data);
    }

    public void Fill(float value)
    {
        Array.Fill(Data, value);
    }

    public void Print(string name)
    {
        Console.WriteLine($"{name} = [");

        for (int i = 0; i < Rows; i++)
        {
            for (int j = 0; j < Columns; j++)
            {
                Console.Write($"{At(i, j),8:F4}");
            }

            Console.WriteLine();
        }

        Console.WriteLine("]");
    }

    public void Clip(float min, float max)
    {
        for (int i = 0; i < Data.Length; i++)
        {
            Data[i] = Math.Clamp(Data[i], min, max);
        }
    }
}