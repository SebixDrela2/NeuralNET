using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public sealed record class DenseHyperParameters(
    NeuralMatrix Weights,
    NeuralMatrix Biases,
    NeuralMatrix PreAct,
    NeuralMatrix PostAct) : IDisposable
{
    public void Dispose()
    {
        Weights.Dispose();
        Biases.Dispose();
        PreAct.Dispose();
        PostAct.Dispose();
    }
}
