using NeutralNET.Matrices;
using System;
using System.Runtime.CompilerServices;

namespace NeutralNET.Activation;

using static LocalScope;

public delegate void ActivationFunction(NeuralMatrix matrix);
public delegate float DerivativeFunction(float activation);

public record struct ActivationFunctionCollection(ActivationFunctionPair Hidden, ActivationFunctionPair Output);
public record struct ActivationFunctionPair(ActivationFunction Activation, DerivativeFunction Derivative)
{
    public static implicit operator ActivationFunctionPair(ActivationType x) => new(GetActivation(x), GetDerivative(x));
}

public class ActivationSelector
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ActivationFunction GetActivation(ActivationType type) => LocalScope.GetActivation(type);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DerivativeFunction GetDerivative(ActivationType type) => LocalScope.GetDerivative(type);
}

file static class LocalScope
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ActivationFunction GetActivation(ActivationType type)
    {
        return type switch
        {
            ActivationType.ReLU => ActivationFunctions.ApplyReLUVectorized,
            ActivationType.LeakyReLU => ActivationFunctions.ApplyLeakyReLUVectorized,
            ActivationType.Sigmoid => ActivationFunctions.ApplySigmoidVectorized,
            ActivationType.Tanh => ActivationFunctions.ApplyTanhVectorized,
            ActivationType.Identity => ActivationFunctions.ApplyLinearVectorized,
            _ => throw new ArgumentOutOfRangeException(nameof(type), $"Unknown activation type: {type}")
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static DerivativeFunction GetDerivative(ActivationType type)
    {
        return type switch
        {
            ActivationType.ReLU => activation => activation > 0 ? 1f : 0f,
            ActivationType.LeakyReLU => activation => activation > 0 ? 1f : 0.01f,
            ActivationType.Sigmoid => activation => Math.Max(activation * (1 - activation), 0.01f), // CLIP IT
            ActivationType.Tanh => activation => Math.Max(1 - activation * activation, 0.01f),
            ActivationType.Identity => activation => 1,
            _ => throw new ArgumentOutOfRangeException(nameof(type), $"Unknown activation type: {type}")
        };
    }
}
