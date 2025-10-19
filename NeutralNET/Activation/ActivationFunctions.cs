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

        var one = Vector512<float>.One;

        for (; ptr != end; ptr += NeuralMatrix.Alignment)
        {
            var vec = Vector512.LoadAligned(ptr);
            var sigmoid = Avx512F.Divide(one, Avx512F.Add(one, Vector512.Exp(Avx512F.Multiply(vec, Vector512.Create(-1.0f)))));
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
            Vector512<float> one = Vector512.Create(1.0f);
            Vector512<float> two = Vector512.Create(2.0f);

            for (; ptr != end; ptr += NeuralMatrix.Alignment)
            {
                var x = Vector512.LoadAligned(ptr);
                var exp2x = Vector512.Exp(Avx512F.Multiply(x, two));
                var tanh = Avx512F.Divide(Avx512F.Subtract(exp2x, one), Avx512F.Add(exp2x, one));
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
        Vector512<float> zero = Vector512<float>.Zero;

        for (; ptr != end; ptr += NeuralMatrix.Alignment)
        {
            var vec = Vector512.LoadAligned(ptr);
            vec = Avx512F.Max(vec, zero);
            vec.StoreAligned(ptr);
        }
    }

    public static void ApplyLeakyReLUVectorized(NeuralMatrix matrix)
    => ApplyLeakyReLUVectorized(matrix.Pointer, matrix.AllocatedLength);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void ApplyLeakyReLUVectorized(float* ptr, int allocatedLength)
    {
        float* end = ptr + allocatedLength;
        Vector512<float> zero = Vector512<float>.Zero;
        Vector512<float> alpha = Vector512.Create(0.01f);

        for (; ptr != end; ptr += NeuralMatrix.Alignment)
        {
            var vec = Vector512.LoadAligned(ptr);
            var mask = Avx512F.CompareLessThan(vec, zero);
            var negPart = Avx512F.Multiply(vec, alpha);
            var posPart = Avx512F.Max(vec, zero);         
            vec = Avx512F.BlendVariable(posPart, negPart, mask);
            vec.StoreAligned(ptr);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplyLinearVectorized(NeuralMatrix matrix)
        => ApplyLinearVectorized(matrix.Pointer, matrix.AllocatedLength);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplyLinearVectorized(float* ptr, int allocatedLength)
    {
        // Identity, do nothing
        // no op
    }
}
