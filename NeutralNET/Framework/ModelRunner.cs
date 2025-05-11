using NeutralNET.Matrices;

namespace NeutralNET.Framework;

public interface IModelRunner
{
    NeuralMatrix Input { get; init; }
    Func<NeuralMatrix> Forward { get; init; }
}

public class ModelRunner : IModelRunner
{
    public required NeuralMatrix Input { get; init; }
    public required Func<NeuralMatrix> Forward { get; init; }
}
