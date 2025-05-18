using NeutralNET.Matrices;
using System;
using System.Runtime.CompilerServices;

namespace NeutralNET.Activation;

public class ActivationSelector
{
    public delegate void ActivationFunction(NeuralMatrix matrix);
    public delegate float DerivativeFunction(float activation);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ActivationFunction GetActivation(ActivationType type)
    {
        return type switch
        {
            ActivationType.ReLU => ActivationFunctions.ApplyReLUVectorized,
            ActivationType.LeakyReLU => ActivationFunctions.ApplyLeakyReLUVectorized,
            ActivationType.Sigmoid => ActivationFunctions.ApplySigmoidVectorized,
            ActivationType.Tanh => ActivationFunctions.ApplyTanhVectorized,
            _ => throw new ArgumentOutOfRangeException(nameof(type), $"Unknown activation type: {type}")
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public DerivativeFunction GetDerivative(ActivationType type, float leakyAlpha = 0.01f)
    {
        return type switch
        {
            ActivationType.ReLU => activation => activation > 0 ? 1f : 0f,
            ActivationType.LeakyReLU => activation => activation > 0 ? 1f : leakyAlpha,
            ActivationType.Sigmoid => activation => Math.Max(activation * (1 - activation), 0.01f), // CLIP IT
            ActivationType.Tanh => activation => Math.Max(1 - activation * activation, 0.01f),
            _ => throw new ArgumentOutOfRangeException(nameof(type), $"Unknown activation type: {type}")
        };
    }
}
