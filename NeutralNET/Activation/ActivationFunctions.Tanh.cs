using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;
using NeutralNET.Matrices;

namespace NeutralNET.Activation;

partial class ActivationFunctions
{
    public sealed class Tanh : IActivationFunction<Tanh>
    {
        public static readonly Tanh Instance = new();
        private Tanh() { }

        public static ActivationType Type => ActivationType.Tanh;

        [MethodImpl(Inline)]
        public static unsafe void Activation(float* ptr, (int Rows, int Cols, int Stride) size)
        {
            int allocatedLength = size.Stride * size.Rows;
            float* end = ptr + allocatedLength;

            if (Avx2.IsSupported) // ?
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

        [MethodImpl(Inline)]
        public static float Derivative(float activation) => float.Max(1 - (activation * activation), 0.01f);
    }
}
