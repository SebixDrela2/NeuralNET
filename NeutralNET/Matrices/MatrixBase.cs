using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace NeutralNET.Matrices;

public abstract unsafe class MatrixBase : IDisposable
{
    private const int Alignment = 8;
    private const int AlignmentMask = Alignment - 1;
    private const int ByteAlignment = Alignment * sizeof(float);
    private const int ByteAlignmentMask = ByteAlignment - 1;

    public static int AllocCounter = 0;
    public static Dictionary<string, StackTrace> Traces = [];
    public static HashSet<(int Width, int Height)> Sizes = [];

    public readonly float* Pointer;

    public int Rows;

    //Obsolete    
    public int ColumnsStride;
    public int UsedColumns;
    public int LogicalLength;
    public int AllocatedLength;
    public uint[] StrideMasks;
    public Span<float> SpanWithGarbage => new(Pointer, AllocatedLength);  

    public MatrixBase(int rows, int columns)
    {
        ColumnsStride = (columns + AlignmentMask) & ~AlignmentMask;

        Rows = rows;
        UsedColumns = columns;

        LogicalLength = Rows * UsedColumns;
        AllocatedLength = Rows * ColumnsStride;

        nuint byteCount = ((nuint)(AllocatedLength * sizeof(float)) + ByteAlignmentMask) & (~(uint)ByteAlignmentMask);      

        Pointer = (float*)NativeMemory.AlignedAlloc(byteCount, ByteAlignment);
        var strideMask = new uint[Vector256<float>.Count];
        var computation = (UsedColumns & AlignmentMask);
        computation = computation is 0 ? Alignment : computation;

        for (var i = 0; i < computation; ++i)
        {
            strideMask[i] = ~0u;
        }

        StrideMasks = strideMask;
        SpanWithGarbage.Clear();
    }

    public void Dispose() => NativeMemory.AlignedFree(Pointer);
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ApplySigmoidVectorized()
    {
        float* ptr = Pointer;
        float* end = ptr + AllocatedLength;

        var one = Vector256<float>.One;

        for (; ptr != end ; ptr += Vector256<float>.Count)
        {
            var vec = Vector256.LoadAligned(ptr);
            var sigmoid = Avx.Divide(one, Avx.Add(one, Vector256.Exp(Avx.Multiply(vec, Vector256.Create(-1.0f)))));
            sigmoid.StoreAligned(ptr);
        }

        //for (; i < SpanWithGarbage.Length; i++)
        //{
        //    Unsafe.Add(ref ptr, i) = 1.0f / (1.0f + float.Exp(-Unsafe.Add(ref ptr, i)));
        //}
    }

    //[MethodImpl(MethodImplOptions.AggressiveInlining)]
    //public void ApplyTanhVectorized()
    //{
    //    Span<float> data = Span;

    //    if (Avx.IsSupported)
    //    {
    //        Vector256<float> one = Vector256.Create(1.0f);
    //        Vector256<float> two = Vector256.Create(2.0f);

    //        int i = 0;
    //        for (; i <= data.Length - Vector256<float>.Count; i += Vector256<float>.Count)
    //        {
    //            Vector256<float> x = Vector256.LoadUnsafe(ref data[i]);
    //            Vector256<float> exp2x = Vector256.Exp(Avx.Multiply(x, two));
    //            Vector256<float> tanh = Avx.Divide(
    //                Avx.Subtract(exp2x, one),
    //                Avx.Add(exp2x, one)
    //            );
    //            tanh.StoreUnsafe(ref data[i]);
    //        }

    //        for (; i < data.Length; i++)
    //        {
    //            data[i] = MathF.Tanh(data[i]);
    //        }
    //    }
    //    else
    //    {
    //        for (int i = 0; i < data.Length; i++)
    //        {
    //            data[i] = MathF.Tanh(data[i]);
    //        }
    //    }
    //}

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ApplyReLUVectorized()
    {
        float* ptr = Pointer;
        float* end = ptr + AllocatedLength;

        Vector256<float> zero = Vector256<float>.Zero;
        Vector256<float> alpha = Vector256.Create(0.01f);

        for (; ptr != end; ptr += Vector256<float>.Count)
        {
            Vector256<float> vec = Vector256.LoadAligned(ptr);
            Vector256<float> mask = Avx.CompareGreaterThan(vec, zero);
            Vector256<float> scaled = Avx.Multiply(vec, alpha);
            Vector256<float> result = Avx.BlendVariable(scaled, vec, mask);

            result.StoreAligned(ptr);
        }

        //for (; i < dataSpan.Length; i++)
        //{
        //    dataSpan[i] = (dataSpan[i] > 0) ? dataSpan[i] : (0.01f * dataSpan[i]);
        //}
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
