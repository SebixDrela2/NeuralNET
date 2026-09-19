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

            for (float* ptr = data, end = ptr + allocatedLength; ptr < end; ptr += size.Stride)
            {
                // Find max for numerical stability
                float max = ptr[0];
                for (int x = 1; x < size.Cols; ++x) max = float.Max(max, ptr[x]);

                // Compute exp(x - max) and sum
                float sum = 0;
                for (int x = 0; x < size.Cols; ++x) sum += ptr[x] = float.Exp(ptr[x] - max); // slow AF, maybe Exp2?

                // Normalize
                float invSum = float.ReciprocalEstimate(sum);
                for (int x = 0; x < size.Cols; ++x) ptr[x] *= invSum;
            }
        }

        [MethodImpl(Inline)]
        public static float Derivative(float activation) => 1;
    }
}
