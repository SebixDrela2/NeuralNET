namespace NeutralNET.Activation;

public interface IActivationFunction
{
    ActivationType ActivationType { get; }
    void ApplyActivationVectorized(float* ptr, (int Rows, int Cols, int Stride) size);
    float GetDerivative(float activation);
}

public interface IActivationFunction<TSelf> : IActivationFunction
    where TSelf : IActivationFunction<TSelf>
{
    static abstract ActivationType Type { get; }
    static abstract void Activation(float* ptr, (int Rows, int Cols, int Stride) size);
    static abstract float Derivative(float activation);

    ActivationType IActivationFunction.ActivationType => TSelf.Type;
    unsafe void IActivationFunction.ApplyActivationVectorized(float* ptr, (int Rows, int Cols, int Stride) size) => TSelf.Activation(ptr, size);
    float IActivationFunction.GetDerivative(float activation) => TSelf.Derivative(activation);
}
