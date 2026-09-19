using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using NeutralNET.Matrices;

namespace NeutralNET.Activation;

partial class ActivationFunctions
{
    public sealed class ReLU : IActivationFunction<ReLU>
    {
        public static readonly ReLU Instance = new();
        private ReLU() { }

        public static ActivationType Type => ActivationType.ReLU;

        [MethodImpl(Inline)]
        public static unsafe void Activation(float* ptr, (int Rows, int Cols, int Stride) size)
        {
            int allocatedLength = size.Stride * size.Rows;

            float* end = ptr + allocatedLength;
            Vector512<float> zero = Vector512<float>.Zero;

            for (; ptr != end; ptr += NeuralMatrix.Alignment)
            {
                var vec = Vector512.LoadAligned(ptr);
                vec = Avx512F.Max(vec, zero);
                vec.StoreAligned(ptr);
            }
        }

        [MethodImpl(Inline)]
        public static float Derivative(float activation) => activation > 0 ? 1 : 0;
    }
}
