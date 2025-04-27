using NeutralNET.Stuff;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Runtime.InteropServices;

namespace NeutralNET.Matrices;

public class Matrix(int rows, int columns) : MatrixBase(rows, columns)
{       
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void DotVectorized(Matrix other, Matrix result)
    {
        int inFeatures = Columns;
        int outFeatures = other.Rows;
        int batchSize = Rows;
        int vecSize = Vector256<float>.Count;
        
        for (int row = 0; row < batchSize; row++)
        {
            Span<float> inputRow = GetRowSpan(row);
            Span<float> resultRow = result.GetRowSpan(row);

            for (int neuronIdx = 0; neuronIdx < outFeatures; neuronIdx++)
            {
                Span<float> weights = other.GetRowSpan(neuronIdx);

                float sum = 0f;
                int k = 0;

                if (Avx.IsSupported)
                {
                    Vector256<float> sumVec = Vector256<float>.Zero;
                    int vectorizable = inFeatures - (inFeatures % vecSize);

                    for (; k < vectorizable; k += vecSize)
                    {
                        var inputVec = Vector256.LoadUnsafe(ref inputRow[k]);
                        var weightVec = Vector256.LoadUnsafe(ref weights[k]);
                        sumVec = Avx.Add(sumVec, Avx.Multiply(inputVec, weightVec));
                    }

                    sum += Vector256.Sum(sumVec);
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
        var span = Span;
        ref float ptr = ref MemoryMarshal.GetReference(span);
        int i = 0;

        var minVec = Vector256.Create(min);
        var maxVec = Vector256.Create(max);
        while (i <= span.Length - Vector256<float>.Count)
        {
            var vec = Vector256.LoadUnsafe(ref ptr, (nuint)i);
            vec = Avx.Min(Avx.Max(vec, minVec), maxVec);
            Vector256.StoreUnsafe(vec, ref ptr, (nuint)i);
            i += Vector256<float>.Count;
        }

        for (; i < span.Length; i++)
        {
            ptr = Math.Clamp(ptr, min, max);
        }
    }
}