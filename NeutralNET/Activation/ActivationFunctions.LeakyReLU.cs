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

            if (Avx512F.IsSupported)
            {
                Vector512<float> zero = Vector512<float>.Zero;
                Vector512<float> alpha = Vector512.Create(Alpha);

                for (; ptr != end; ptr += Vector512<float>.Count)
                {
                    var vec = Vector512.LoadAligned(ptr);
                    var mask = Avx512F.CompareLessThan(vec, zero);
                    var negPart = Avx512F.Multiply(vec, alpha);
                    var posPart = Avx512F.Max(vec, zero);
                    vec = Avx512F.BlendVariable(posPart, negPart, mask);
                    vec.StoreAligned(ptr);
                }
            }
            else if (Avx2.IsSupported)
            {
                Vector256<float> zero = Vector256<float>.Zero;
                Vector256<float> alpha = Vector256.Create(Alpha);
                int vectorSize = Vector256<float>.Count; // 8 floats

                for (; ptr != end; ptr += vectorSize)
                {
                    var vec = Vector256.Load(ptr);
                    var mask = Vector256.LessThan(vec, zero);
                    var negPart = Vector256.Multiply(vec, alpha);
                    var posPart = Vector256.Max(vec, zero);
                    var result = Vector256.ConditionalSelect(mask, negPart, posPart);
                    result.Store(ptr);
                }
            }
            else
            {
                for (; ptr != end; ptr++)
                {
                    float val = *ptr;
                    *ptr = val < 0f ? val * Alpha : val;
                }
            }
        }

        [MethodImpl(Inline)]
        public static float Derivative(float activation) => activation > 0 ? 1 : Alpha;
    }
}
