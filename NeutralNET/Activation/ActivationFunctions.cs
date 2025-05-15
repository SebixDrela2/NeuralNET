using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using NeutralNET.Matrices;

namespace NeutralNET.Activation;

public static unsafe class ActivationFunctions
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplySigmoidVectorized(NeuralMatrix matrix)
        => ApplySigmoidVectorized(matrix.Pointer, matrix.AllocatedLength);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplySigmoidVectorized(float* ptr, int allocatedLength)
    {
        float* end = ptr + allocatedLength;

        var one = Vector256<float>.One;

        for (; ptr != end; ptr += Vector256<float>.Count)
        {
            var vec = Vector256.LoadAligned(ptr);
            var sigmoid = Avx.Divide(one, Avx.Add(one, Vector256.Exp(Avx.Multiply(vec, Vector256.Create(-1.0f)))));
            sigmoid.StoreAligned(ptr);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplyTanhVectorized(NeuralMatrix matrix)
        => ApplyTanhVectorized(matrix.Pointer, matrix.AllocatedLength);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplyTanhVectorized(float* ptr, int allocatedLength)
    {
        float* end = ptr + allocatedLength;

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
    public static void ApplyReLUVectorized(NeuralMatrix matrix)
        => ApplyReLUVectorized(matrix.Pointer, matrix.AllocatedLength);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplyReLUVectorized(float* ptr, int allocatedLength)
    {
        float* end = ptr + allocatedLength;
        Vector256<float> zero = Vector256<float>.Zero;

        for (; ptr != end; ptr += Vector256<float>.Count)
        {
            var vec = Vector256.LoadAligned(ptr);
            vec = Avx.Max(vec, zero);  // Standard ReLU
            vec.StoreAligned(ptr);
        }
    }
}
