using NeutralNET.Matrices;

namespace NeutralNET.Activation;

public static partial class ActivationFunctions
{
    extension<T>(T self)
        where T : class, IActivationFunction
    {
        [MethodImpl(Inline)]
        public unsafe void ApplyActivationVectorized(NeuralMatrix matrix) => self.ApplyActivationVectorized(matrix.Pointer, (matrix.Rows, matrix.UsedColumns, matrix.ColumnsStride));
    }

    [MethodImpl(Inline)]
    public static IActivationFunction Resolve(ActivationType type) => type switch
    {
        ActivationType.Identity => Identity.Instance,
        ActivationType.ReLU => ReLU.Instance,
        ActivationType.LeakyReLU => LeakyReLU.Instance,
        ActivationType.Sigmoid => Sigmoid.Instance,
        ActivationType.Tanh => Tanh.Instance,
        ActivationType.Softmax => SoftMax.Instance,
        _ => throw new ArgumentOutOfRangeException(nameof(type), $"Unknown activation type: {type}")
    };
}
