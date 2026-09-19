using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using NeutralNET.Matrices;

namespace NeutralNET.Activation;

partial class ActivationFunctions
{
    public sealed class Sigmoid : IActivationFunction<Sigmoid>
    {
        public static readonly Sigmoid Instance = new();
        private Sigmoid() { }

        public static ActivationType Type => ActivationType.Sigmoid;

        [MethodImpl(Inline)]
        public static unsafe void Activation(float* ptr, (int Rows, int Cols, int Stride) size)
        {
            int allocatedLength = size.Stride * size.Rows;
            float* end = ptr + allocatedLength;

            var one = Vector512<float>.One;

            for (; ptr != end; ptr += NeuralMatrix.Alignment)
            {
                var vec = Vector512.LoadAligned(ptr);
                var sigmoid = Avx512F.Divide(one, Avx512F.Add(one, Vector512.Exp(Avx512F.Multiply(vec, Vector512.Create(-1.0f)))));
                sigmoid.StoreAligned(ptr);
            }
        }

        [MethodImpl(Inline)]
        public static float Derivative(float activation) => float.Max(activation * (1 - activation), 0.01f);
    }
}
