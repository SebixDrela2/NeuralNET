using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using NeutralNET.Matrices;

namespace NeutralNET.Activation;

partial class ActivationFunctions
{
    public sealed class LeakyReLU : IActivationFunction<LeakyReLU>
    {
        public const float Alpha = 0.01f;

        public static readonly LeakyReLU Instance = new();
        private LeakyReLU() { }

        public static ActivationType Type => ActivationType.LeakyReLU;

        [MethodImpl(Inline)]
        public static unsafe void Activation(float* ptr, (int Rows, int Cols, int Stride) size)
        {
            int allocatedLength = size.Stride * size.Rows;
            float* end = ptr + allocatedLength;

            Vector512<float> zero = Vector512<float>.Zero;
            Vector512<float> alpha = Vector512.Create(Alpha);

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

        [MethodImpl(Inline)]
        public static float Derivative(float activation) => activation > 0 ? 1 : Alpha;
    }
}
