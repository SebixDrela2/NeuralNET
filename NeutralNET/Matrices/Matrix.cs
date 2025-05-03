using NeutralNET.Stuff;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Matrices;

public unsafe class Matrix(int rows, int columns) : MatrixBase(rows, columns)
{   
    // TODO: UNVECTORIZE
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void DotVectorized(Matrix other, Matrix result)
    {
        int inFeatures = UsedColumns;
        int outFeatures = other.Rows;
        int batchSize = Rows;
        int vecSize = Vector256<float>.Count;

        var inputRow = GetMatrixRow(0);
        var resultRow = result.GetMatrixRow(0);

        for (int row = 0; row < batchSize; row++, ++inputRow, ++resultRow)
        {
            var weights = other.GetMatrixRow(0);

            for (int neuronIdx = 0; neuronIdx < outFeatures; neuronIdx++, ++weights)
            {               
                var sum = 0f;
                var k = 0;

                var sumVec = Vector256<float>.Zero;
                var vectorizable = inFeatures - (inFeatures % vecSize);

                for (; k < vectorizable; k += vecSize)
                {
                    var inputVec = inputRow.LoadVectorAligned(k);
                    var weightVec = weights.LoadVectorAligned(k);
                    sumVec = Fma.MultiplyAdd(inputVec, weightVec, sumVec);
                }

                sum += Vector256.Sum(sumVec);

                for (; k < inFeatures; k++)
                {
                    sum += inputRow.Span[k] * weights.Span[k];
                }

                resultRow.Span[neuronIdx] = sum;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<float> GetRowSpan(int row) => SpanWithGarbage.Slice(row * ColumnsStride, UsedColumns);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public MatrixRow GetMatrixRow(int row)
    {
        float* rowPtr = row * ColumnsStride + Pointer;

        return new MatrixRow(rowPtr, UsedColumns, ColumnsStride);
    }

    public void CopyRowFrom(Matrix other, int row)
    {
        other.GetRowSpan(row).CopyTo(GetRowSpan(row));
    }

    public void CopyDataFrom(Matrix other)
    {
        NativeMemory.Copy(other.Pointer, Pointer, (nuint)AllocatedLength * sizeof(float));
    }

    public Matrix SplitStart(int column)
    {
        var result = new Matrix(Rows, column);

        for (int i = 0; i < Rows; i++)
        {
            GetRowSpan(i).Slice(0, column).CopyTo(result.GetRowSpan(i));
        }

        return result;
    }

    public Matrix SplitEnd(int column)
    {
        var result = new Matrix(Rows, 1);

        for (int i = 0; i < Rows; i++)
        {
            result.Set(i, 0, GetRowSpan(i)[column - 1]);
        }

        return result;
    }

    // TODO: OPTIMIZE
    public void Sum(Matrix other)
    {
        if (Rows != other.Rows || UsedColumns != other.UsedColumns)
        {
            throw new ArgumentException("Dimension mismatch");
        }

        var span = SpanWithGarbage;
        var otherSpan = other.SpanWithGarbage;

        for (int i = 0; i < span.Length; i++)
        {
            span[i] += otherSpan[i];
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

    public void Clear()
    {
        NativeMemory.Clear(Pointer, (nuint)AllocatedLength * sizeof(float));
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
    }

    public void Clamp(float min, float max)
    {
        int vectorSize = Vector256<float>.Count;
        int i = 0;

        float* ptr = Pointer;
        float* end = ptr + AllocatedLength;

        var minVec = Vector256.Create(min);
        var maxVec = Vector256.Create(max);

        for (; ptr != end; ptr += Vector256<float>.Count)
        {
            var vec = Vector256.LoadAligned(ptr);
            vec = Vector256.ClampNative(vec, minVec, maxVec);         
            vec.StoreAligned(ptr);
        }

        //for (; i < span.Length; i++)
        //{
        //    span[i] = Math.Clamp(span[i], min, max);
        //}
    }
}