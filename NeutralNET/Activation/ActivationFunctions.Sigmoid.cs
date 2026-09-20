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

            if (Avx512F.IsSupported)
            {
                var one = Vector512<float>.One;

                for (; ptr != end; ptr += Vector512<float>.Count)
                {
                    var vec = Vector512.LoadAligned(ptr);
                    var sigmoid = Avx512F.Divide(one, Avx512F.Add(one, Vector512.Exp(Avx512F.Multiply(vec, Vector512.Create(-1.0f)))));
                    sigmoid.StoreAligned(ptr);
                }
            }
            else if (Avx2.IsSupported)
            {
                var one = Vector256<float>.One;
                int vectorSize = Vector256<float>.Count; // 8 floats

                for (; ptr != end; ptr += vectorSize)
                {
                    var vec = Vector256.LoadAligned(ptr);
                    var sigmoid = Vector256.Divide(one, Vector256.Add(one, Vector256.Exp(Vector256.Multiply(vec, Vector256.Create(-1.0f)))));
                    sigmoid.StoreAligned(ptr);
                }
            }
            else
            {
                for (; ptr != end; ptr++)
                {
                    float val = *ptr;
                    *ptr = 1.0f / (1.0f + MathF.Exp(-val));
                }
            }
        }

        [MethodImpl(Inline)]
        public static float Derivative(float activation) => float.Max(activation * (1 - activation), 0.01f);
    }
}
