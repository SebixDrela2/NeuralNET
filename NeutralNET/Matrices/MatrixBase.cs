using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Diagnostics;

namespace NeutralNET.Matrices;

public abstract class MatrixBase
{
    public static int AllocCounter = 0;
    public static Dictionary<string, StackTrace> Traces = [];
    public static HashSet<(int Width, int Height)> Sizes = [];

    public int Rows { get; }
    public int Columns { get; }
    public float[] Data { get; set; }
    public Span<float> Span => Data;
    public float FirstElement => Data[0];

    public MatrixBase(int rows, int columns)
    {
        Rows = rows;
        Columns = columns;

        Data = new float[rows * columns];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ApplySigmoid()
    {
        for (int i = 0; i < Data.Length; i++)
        {
            Data[i] = 1f / (1f + float.Exp(-Data[i]));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ApplyTanh()
    {
        Span<float> data = Data.AsSpan();

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
        Span<float> dataSpan = Data.AsSpan();
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
