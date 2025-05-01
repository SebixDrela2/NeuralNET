using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace NeutralNET.Matrices;

public abstract unsafe class MatrixBase : IDisposable
{
    public static int AllocCounter = 0;
    public static Dictionary<string, StackTrace> Traces = [];
    public static HashSet<(int Width, int Height)> Sizes = [];

    protected readonly float* _alignedData;
    private bool _disposed;

    public int Rows;

    //Obsolete
    protected int Columns;
    public int ColumnsStride  => Columns;
    public int UsedColumns => Columns;
    public int Count;

    public Span<float> Span => new(_alignedData, Count);
    public MatrixElement Pointer => new(_alignedData, Count);

    public MatrixBase(int rows, int columns)
    {
        Rows = rows;
        Columns = columns;
        Count = Columns * Rows;

        nuint byteCount = (nuint)(rows * columns * sizeof(float));
        const uint alignment = 32;
        _alignedData = (float*)NativeMemory.AlignedAlloc(byteCount, alignment);

        Span.Clear();
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
        {
            return;
        }

        NativeMemory.AlignedFree(_alignedData);
        _disposed = true;
    }    

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ApplySigmoidVectorized()
    {
        ref float ptr = ref MemoryMarshal.GetReference(Span);
        int i = 0;

        var one = Vector256.Create(1.0f);
        var half = Vector256.Create(0.5f);

        while (i <= Span.Length - Vector256<float>.Count)
        {
            var vec = Vector256.LoadUnsafe(ref ptr, (nuint)i);
            var sigmoid = Avx.Divide(one, Avx.Add(one, Vector256.Exp(Avx.Multiply(vec, Vector256.Create(-1.0f)))));
            sigmoid.StoreUnsafe(ref ptr, (nuint)i);
            i += Vector256<float>.Count;
        }

        for (; i < Span.Length; i++)
        {
            Unsafe.Add(ref ptr, i) = 1.0f / (1.0f + float.Exp(-Unsafe.Add(ref ptr, i)));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ApplyTanhVectorized()
    {
        Span<float> data = Span;

        if (Avx.IsSupported)
        {
            Vector256<float> one = Vector256.Create(1.0f);
            Vector256<float> two = Vector256.Create(2.0f);

            int i = 0;
            for (; i <= data.Length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                Vector256<float> x = Vector256.LoadUnsafe(ref data[i]);
                Vector256<float> exp2x = Vector256.Exp(Avx.Multiply(x, two));
                Vector256<float> tanh = Avx.Divide(
                    Avx.Subtract(exp2x, one),
                    Avx.Add(exp2x, one)
                );
                tanh.StoreUnsafe(ref data[i]);
            }

            for (; i < data.Length; i++)
            {
                data[i] = MathF.Tanh(data[i]);
            }
        }
        else
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = MathF.Tanh(data[i]);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ApplyReLUVectorized()
    {
        Span<float> dataSpan = Span;
        int i = 0;

        Vector256<float> zero = Vector256<float>.Zero;
        Vector256<float> alpha = Vector256.Create(0.01f);

        for (; i <= dataSpan.Length - Vector256<float>.Count; i += Vector256<float>.Count)
        {
            Vector256<float> vec = Vector256.LoadUnsafe(ref dataSpan[i]);
            Vector256<float> mask = Avx.CompareGreaterThan(vec, zero);
            Vector256<float> scaled = Avx.Multiply(vec, alpha);
            Vector256<float> result = Avx.BlendVariable(scaled, vec, mask);

            result.StoreUnsafe(ref dataSpan[i]);
        }

        for (; i < dataSpan.Length; i++)
        {
            dataSpan[i] = (dataSpan[i] > 0) ? dataSpan[i] : (0.01f * dataSpan[i]);
        }
    }

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
}
