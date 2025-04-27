using NeutralNET.Stuff;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Matrices;

public unsafe class Matrix(int rows, int columns) : MatrixBase(rows, columns), IDisposable
{
    private sealed class UnmanagedMemoryManager : MemoryManager<float>
    {
        private readonly float* _pointer;
        private readonly int _length;

        public UnmanagedMemoryManager(float* pointer, int length)
        {
            _pointer = pointer;
            _length = length;
        }

        public override Span<float> GetSpan() => new Span<float>(_pointer, _length);

        public override MemoryHandle Pin(int elementIndex = 0) =>
            new MemoryHandle(_pointer + elementIndex);

        public override void Unpin() { }

        protected override void Dispose(bool disposing) { }
    }

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
    public Span<float> GetRowSpan(int row) => Span.Slice(row * Columns, Columns);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]   
    public Memory<float> GetRowMemory(int row)
    {
        unsafe
        {
            float* rowPtr = _alignedData + row * Columns;
            return new UnmanagedMemoryManager(rowPtr, Columns).Memory;
        }
    }
    public void CopyRowFrom(Matrix other, int row)
    {
        other.GetRowSpan(row).CopyTo(GetRowSpan(row));
    }

    public void CopyDataFrom(Matrix other)
    {
        other.Span.CopyTo(Span);
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

    public Matrix Row(int row)
    {
        var result = new Matrix(1, Columns);
        GetRowSpan(row).CopyTo(result.Span);
        return result;
    }

    public void SumVec(Matrix other)
    {
        var span = Span;
        var otherSpan = other.Span;

        int vectorSize = Vector256<float>.Count;
        int i = 0;

        for (; i <= span.Length - vectorSize; i += vectorSize)
        {
            var va = Vector256.LoadAligned((float*)Unsafe.AsPointer(ref span[i]));
            var vb = Vector256.LoadAligned((float*)Unsafe.AsPointer(ref otherSpan[i]));
            var sum = Avx.Add(va, vb);

            sum.StoreAligned((float*)Unsafe.AsPointer(ref span[i]));
        }

        for (; i < span.Length; i++)
        {
            span[i] += otherSpan[i];
        }
    }

    public void Sum(Matrix other)
    {
        if (Rows != other.Rows || Columns != other.Columns)
        {
            throw new ArgumentException("Dimension mismatch");
        }

        var span = Span;
        var otherSpan = other.Span;

        for (int i = 0; i < span.Length; i++)
        {
            span[i] += otherSpan[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ref float At(int row, int column) => ref Span[row * Columns + column];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Set(int row, int column, float value) => Span[row * Columns + column] = value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(int row, int column, float value) => Span[row * Columns + column] += value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Sub(int row, int column, float value) => Span[row * Columns + column] -= value;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Divide(int row, int column, float value) => Span[row * Columns + column] /= value;

    public void Randomize(float low = 0, float high = 1)
    {
        var span = Span;
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = RandomUtils.GetFloat(1) * (high - low) + low;
        }
    }

    public new void Clear()
    {
        var span = Span;
        span.Clear();
    }

    public void Fill(float value)
    {
        Span.Fill(value);
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

    public void Clip(float min, float max)
    {
        var span = Span;
        int vectorSize = Vector256<float>.Count;
        int i = 0;

        var minVec = Vector256.Create(min);
        var maxVec = Vector256.Create(max);

        for (; i <= span.Length - vectorSize; i += vectorSize)
        {
            var vec = Vector256.LoadAligned((float*)Unsafe.AsPointer(ref span[i]));
            vec = Avx.Min(Avx.Max(vec, minVec), maxVec);
            vec.StoreAligned((float*)Unsafe.AsPointer(ref span[i]));
        }

        for (; i < span.Length; i++)
        {
            span[i] = Math.Clamp(span[i], min, max);
        }
    }
}