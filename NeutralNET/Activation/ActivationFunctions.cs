using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using NeutralNET.Matrices;

namespace NeutralNET.Activation;

public static unsafe class ActivationFunctions
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplySigmoidVectorized(NeuralMatrix matrix)
        => ApplySigmoidVectorized(matrix.Pointer, matrix.UnsafeSize);

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
        => ApplyTanhVectorized(matrix.Pointer, matrix.UnsafeSize);
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
        => ApplyReLUVectorized(matrix.Pointer, matrix.UnsafeSize);
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
    => ApplyLeakyReLUVectorized(matrix.Pointer, matrix.UnsafeSize);

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
        => ApplyLinearVectorized(matrix.Pointer, matrix.UnsafeSize);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplyLinearVectorized(float* ptr, int allocatedLength)
    {
        // Identity, do nothing
        // no op
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ApplySoftmaxVectorized(NeuralMatrix matrix)
    {
        float* ptr = matrix.Pointer;
        int rows = matrix.Rows;
        int cols = matrix.UsedColumns;
        int stride = matrix.ColumnsStride;   // ← THE FIX

        for (int r = 0; r < rows; r++)
        {
            float* row = ptr + r * stride;   // ← Use stride, not cols

            // Find max for numerical stability
            float max = row[0];
            for (int c = 1; c < cols; c++)
                if (row[c] > max) max = row[c];

            // Compute exp(x - max) and sum
            float sum = 0f;
            for (int c = 0; c < cols; c++)
            {
                row[c] = MathF.Exp(row[c] - max);
                sum += row[c];
            }

            // Normalize
            float invSum = 1.0f / sum;
            for (int c = 0; c < cols; c++)
                row[c] *= invSum;
        }
    }
}
