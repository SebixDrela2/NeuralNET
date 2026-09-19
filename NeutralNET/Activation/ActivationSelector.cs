using NeutralNET.Matrices;

namespace NeutralNET.Activation;

public class ActivationSelector
{
    public delegate void ActivationFunction(NeuralMatrix matrix);
    public delegate float DerivativeFunction(float activation);

    [MethodImpl(Inline)]
    public ActivationFunction GetActivation(ActivationType type)
    {
        return type switch
        {
            ActivationType.ReLU => ActivationFunctions.ReLU.Instance.ApplyActivationVectorized,
            ActivationType.LeakyReLU => ActivationFunctions.LeakyReLU.Instance.ApplyActivationVectorized,
            ActivationType.Sigmoid => ActivationFunctions.Sigmoid.Instance.ApplyActivationVectorized,
            ActivationType.Tanh => ActivationFunctions.Tanh.Instance.ApplyActivationVectorized,
            ActivationType.Identity => ActivationFunctions.Identity.Instance.ApplyActivationVectorized,
            ActivationType.Softmax => ActivationFunctions.SoftMax.Instance.ApplyActivationVectorized,
            _ => throw new ArgumentOutOfRangeException(nameof(type), $"Unknown activation type: {type}")
        };
    }


    [MethodImpl(Inline)]
    public DerivativeFunction GetDerivative(ActivationType type)
    {
        return type switch
        {
            ActivationType.ReLU => ActivationFunctions.ReLU.Derivative,
            ActivationType.LeakyReLU => ActivationFunctions.LeakyReLU.Derivative,
            ActivationType.Sigmoid => ActivationFunctions.Sigmoid.Derivative,
            ActivationType.Tanh => ActivationFunctions.Tanh.Derivative,
            ActivationType.Identity => ActivationFunctions.Identity.Derivative,
            ActivationType.Softmax => ActivationFunctions.SoftMax.Derivative,
            _ => throw new ArgumentOutOfRangeException(nameof(type), $"Unknown activation type: {type}")
        };
    }
}
