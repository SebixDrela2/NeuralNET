namespace NeutralNET.Activation;

partial class ActivationFunctions
{
    public sealed class Identity : IActivationFunction<Identity>
    {
        public static readonly Identity Instance = new();
        private Identity() { }

        public static ActivationType Type => ActivationType.Identity;

        [MethodImpl(Inline)]
        public static void Activation(float* ptr, (int Rows, int Cols, int Stride) size) { /* no-op */ }

        [MethodImpl(Inline)]
        public static float Derivative(float activation) => 1;
    }
}
