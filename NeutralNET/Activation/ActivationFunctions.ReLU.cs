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

            if (Avx512F.IsSupported)
            {
                Vector512<float> zero = Vector512<float>.Zero;

                for (; ptr != end; ptr += Vector512<float>.Count)
                {
                    var vec = Vector512.LoadAligned(ptr);
                    vec = Avx512F.Max(vec, zero);
                    vec.StoreAligned(ptr);
                }
            }
            else if (Avx2.IsSupported)
            {
                Vector256<float> zero = Vector256<float>.Zero;
                int vectorSize = Vector256<float>.Count; // 8 floats

                for (; ptr != end; ptr += vectorSize)
                {
                    var vec = Vector256.LoadAligned(ptr);
                    vec = Vector256.Max(vec, zero);
                    vec.StoreAligned(ptr);
                }
            }
            else
            {
                for (; ptr != end; ptr++)
                {
                    float val = *ptr;
                    *ptr = val < 0f ? 0f : val;
                }
            }
        }

        [MethodImpl(Inline)]
        public static float Derivative(float activation) => activation > 0 ? 1 : 0;
    }
}
