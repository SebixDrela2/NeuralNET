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

            if (Vector512.IsHardwareAccelerated && Avx512F.IsSupported)
            {
                var one = Vector512.Create(1.0f);
                var two = Vector512.Create(2.0f);

                for (; ptr != end; ptr += Vector512<float>.Count)
                {
                    var x = Vector512.LoadAligned(ptr);
                    var exp2x = Vector512.Exp(Vector512.Multiply(x, two));
                    var tanh = Vector512.Divide(Vector512.Subtract(exp2x, one), Vector512.Add(exp2x, one));
                    tanh.StoreAligned(ptr);
                }
            }
            else if (Vector256.IsHardwareAccelerated && Avx2.IsSupported)
            {
                var one = Vector256.Create(1.0f);
                var two = Vector256.Create(2.0f);
                int vectorSize = Vector256<float>.Count; // 8 floats

                for (; ptr != end; ptr += vectorSize)
                {
                    var x = Vector256.LoadAligned(ptr);
                    var exp2x = Vector256.Exp(Vector256.Multiply(x, two));
                    var tanh = Vector256.Divide(Vector256.Subtract(exp2x, one), Vector256.Add(exp2x, one));
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
