using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace NeutralNET.Matrices;

public abstract unsafe class NeuralMatrixBase : IDisposable
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

    public NeuralMatrixBase(int rows, int columns)
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
            //sigmoid = Avx.Multiply(sigmoid, Vector256.Create<float>(2f));
            //sigmoid = Avx.Subtract(sigmoid, one);
            sigmoid.StoreAligned(ptr);
        }

        //for (var i = 0; i < SpanWithGarbage.Length; i++)
        //{
        //    ptr[i] = 1.0f / (1.0f + float.Exp(-ptr[i]));
        //}
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ApplyTanhVectorized()
    {
        float* ptr = Pointer;
        float* end = ptr + AllocatedLength;

        if (Avx2.IsSupported)
        {
            Vector256<float> one = Vector256.Create(1.0f);
            Vector256<float> two = Vector256.Create(2.0f);

            for (; ptr != end; ptr += Vector256<float>.Count)
            {
                var x = Vector256.LoadAligned(ptr);
                var exp2x = Vector256.Exp(Avx.Multiply(x, two));
                var tanh = Avx.Divide(Avx.Subtract(exp2x, one), Avx.Add(exp2x, one));
                tanh.StoreAligned(ptr);
            }
        }
        else
        {
            for (; ptr < end; ptr++)
            {
                *ptr = MathF.Tanh(*ptr);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ApplyReLUVectorized()
    {
        float* ptr = Pointer;
        float* end = ptr + AllocatedLength;
        Vector256<float> zero = Vector256<float>.Zero;

        for (; ptr != end; ptr += Vector256<float>.Count)
        {
            var vec = Vector256.LoadAligned(ptr);
            vec = Avx.Max(vec, zero);  // Standard ReLU
            vec.StoreAligned(ptr);
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
