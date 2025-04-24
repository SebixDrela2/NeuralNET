using NeutralNET.Stuff;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace NeutralNET.Matrices;

public class Matrix
{
    private static int AllocCounter = 0;
    public static Dictionary<string, StackTrace> Traces = [];
    public static HashSet<(int Width, int Height)> Sizes = [];
    
    public int Rows { get; }
    public int Columns { get; }
    public float[] Data { get; set; }
    public Span<float> Span => Data;
    public float FirstElement => Data[0];

    // [Conditional("DEBUG")]
    private static void LogOrigin(int rows, int columns)
    {
        var i = AllocCounter++;
        
        // switch (i)
        // {
        //     case > 1_000_000 when (i % 1_000_000) is not 0:
        //     case > 100_000 when (i % 100_000) is not 0:
        //     case > 10_000 when (i % 10_000) is not 0:
        //     case > 1_000 when (i % 1_000) is not 0:
        //     case > 100 when (i % 100) is not 0:
        //     case > 10 when (i % 10) is not 0:
        //         break;
        //     default:
        //         Console.WriteLine($"NEW ARRAY CREATED {i}");
        //         break;
        // }

        var stack = new StackTrace();
        var frames = string.Join("\n", stack.GetFrames().Reverse()
            .Select(x => $"{x.GetMethod()?.DeclaringType?.FullName}.{x.GetMethod()?.Name}"));
        Traces.TryAdd(frames, stack);

        Sizes.Add((columns, rows));
    }

    public Matrix(int rows, int columns)
    {
        Rows = rows;
        Columns = columns;
        
        Data = new float[rows * columns];
        // LogOrigin(rows, columns);
    }

    public Matrix this[int row] => Row(row);

    public void ApplySigmoid()
    {
        for (int i = 0; i < Data.Length; i++)
            Data[i] = 1f / (1f + float.Exp(-Data[i]));
    }

    public void ApplyReLU()
    {
        for (int i = 0; i < Data.Length; i++)
            Data[i] = float.Max(0.01f * Data[i], Data[i]); // Leaky ReLU
    }

    public void Dot(Matrix other, Matrix result)
    {
        if (Columns != other.Rows)
            throw new ArgumentException("Dimension mismatch");

        //var result = new Matrix(Rows, other.Columns);

        for (int row = 0; row < Rows; row++)
        {
            for (int col = 0; col < other.Columns; col++)
            {
                float sum = 0f;
                for (int k = 0; k < Columns; k++)
                    sum += At(row, k) * other.At(k, col);
                result.Set(row, col, sum);
            }
        }

        //return result;
    }

    public Memory<float> GetRowMemory(int row) => Data.AsMemory(row * Columns, Columns);
    public Span<float> GetRowSpan(int row) => Data.AsSpan(row * Columns, Columns);
    public void CopyRowFrom(Matrix other, int row)
    {
        other.GetRowSpan(row).CopyTo(Span);
    }
    public void CopyDataFrom(Matrix other)
    {
        if (Rows != other.Rows || Columns != other.Columns)
        {
            throw new ArgumentException("Dimension mismatch");
        }

        Buffer.BlockCopy(other.Data, 0, Data, 0, Data.Length * sizeof(float));
    }

    public Matrix SplitStart(int column)
    {
        var result = new Matrix(Rows, column);
        for (int i = 0; i < Rows; i++)
            for (int j = 0; j < column; j++)
                result.Set(i, j, At(i, j));
        return result;
    }

    public Matrix SplitEnd(int column)
    {
        var result = new Matrix(Rows, 1);
        for (int i = 0; i < Rows; i++)
            result.Set(i, 0, At(i, column - 1));
        return result;
    }

    // public IEnumerable<Matrix> BatchAllRows(int batchSize)
    // {
    //     if (batchSize <= 0)
    //     {
    //         throw new ArgumentException("Batch size must be positive");
    //     }

    //     for (int startRow = 0; startRow < Rows; startRow += batchSize)
    //     {
    //         yield return BatchView(startRow, batchSize);
    //     }
    // }

    public Matrix BatchView(int startRow, int rowCount)
    {
        var actualRows = int.Min(rowCount, Rows - startRow);
        var result = new Matrix(actualRows, Columns);
        Buffer.BlockCopy(Data, startRow * Columns * sizeof(float),
                       result.Data, 0,
                       actualRows * Columns * sizeof(float));

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
    public float At(int row, int column) => Data[row * Columns + column];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set(int row, int column, float value) => Data[row * Columns + column] = value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(int row, int column, float value) => Data[row * Columns + column] += value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Sub(int row, int column, float value) => Data[row * Columns + column] -= value;

    public void Divide(int row, int column, float value) => Data[row * Columns + column] /= value;

    public void Randomize(float low = 0, float high = 1)
    {
        for (int i = 0; i < Data.Length; i++)
        {
            Data[i] = RandomUtils.GetFloat(1) * (high - low) + low;
        }
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
    
    public Matrix Reorder(int[] newIndices)
    {
        var result = new Matrix(Rows, Columns);
        for (int newRow = 0; newRow < newIndices.Length; newRow++)
        {
            int originalRow = newIndices[newRow];
            for (int col = 0; col < Columns; col++)
                result.Set(newRow, col, At(originalRow, col));
        }
        return result;
    }
}