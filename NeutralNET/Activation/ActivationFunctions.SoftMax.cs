using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace NeutralNET.Activation;

partial class ActivationFunctions
{
    public sealed class SoftMax : IActivationFunction<SoftMax>
    {
        public static readonly SoftMax Instance = new();
        private SoftMax() { }

        public static ActivationType Type => ActivationType.Softmax;

        [MethodImpl(Inline)]
        public static unsafe void Activation(float* data, (int Rows, int Cols, int Stride) size)
        {
            int allocatedLength = size.Stride * size.Rows;

            if (Vector512.IsHardwareAccelerated && Avx512F.IsSupported)
            {
                for (float* ptr = data, end = ptr + allocatedLength; ptr < end; ptr += size.Stride)
                {
                    float max = ptr[0];
                    for (int y = 1; y < size.Cols; ++y) max = float.Max(max, ptr[y]);

                    float sum = 0;
                    Vector512<float> maxVec = Vector512.Create(max);
                    Vector512<float> sumVec = Vector512<float>.Zero;
                    int x = 0;

                    int vCount = Vector512<float>.Count;
                    for (; x <= size.Cols - vCount; x += vCount)
                    {
                        var vec = Vector512.Load(ptr + x);
                        var expVec = Vector512.Exp(Vector512.Subtract(vec, maxVec));
                        expVec.Store(ptr + x);
                        sumVec = Vector512.Add(sumVec, expVec);
                    }
                    sum = Vector512.Sum(sumVec);

                    for (; x < size.Cols; ++x)
                    {
                        float val = float.Exp(ptr[x] - max);
                        ptr[x] = val;
                        sum += val;
                    }

                    float invSum = float.ReciprocalEstimate(sum);
                    x = 0;
                    for (; x <= size.Cols - vCount; x += vCount)
                    {
                        var vec = Vector512.Load(ptr + x);
                        vec = Vector512.Multiply(vec, Vector512.Create(invSum));
                        vec.Store(ptr + x);
                    }
                    for (; x < size.Cols; ++x)
                    {
                        ptr[x] *= invSum;
                    }
                }
            }
            else if (Avx2.IsSupported)
            {
                for (float* ptr = data, end = ptr + allocatedLength; ptr < end; ptr += size.Stride)
                {
                    float max = ptr[0];
                    for (int y = 1; y < size.Cols; ++y) max = float.Max(max, ptr[y]);

                    float sum = 0;
                    Vector256<float> maxVec = Vector256.Create(max);
                    Vector256<float> sumVec = Vector256<float>.Zero;
                    int x = 0;

                    int vCount = Vector256<float>.Count;
                    for (; x <= size.Cols - vCount; x += vCount)
                    {
                        var vec = Vector256.Load(ptr + x);
                        var expVec = Vector256.Exp(Vector256.Subtract(vec, maxVec));
                        expVec.Store(ptr + x);
                        sumVec = Vector256.Add(sumVec, expVec);
                    }
                    sum = Vector256.Sum(sumVec);

                    for (; x < size.Cols; ++x)
                    {
                        float val = float.Exp(ptr[x] - max);
                        ptr[x] = val;
                        sum += val;
                    }

                    float invSum = float.ReciprocalEstimate(sum);
                    x = 0;
                    for (; x <= size.Cols - vCount; x += vCount)
                    {
                        var vec = Vector256.Load(ptr + x);
                        vec = Vector256.Multiply(vec, Vector256.Create(invSum));
                        vec.Store(ptr + x);
                    }
                    for (; x < size.Cols; ++x)
                    {
                        ptr[x] *= invSum;
                    }
                }
            }
            else
            {
                for (float* ptr = data, end = ptr + allocatedLength; ptr < end; ptr += size.Stride)
                {
                    float max = ptr[0];
                    for (int x = 1; x < size.Cols; ++x) max = float.Max(max, ptr[x]);

                    float sum = 0;
                    for (int x = 0; x < size.Cols; ++x) sum += ptr[x] = float.Exp(ptr[x] - max);

                    float invSum = float.ReciprocalEstimate(sum);
                    for (int x = 0; x < size.Cols; ++x) ptr[x] *= invSum;
                }
            }
        }

        [MethodImpl(Inline)]
        public static float Derivative(float activation) => 1;
    }
}
