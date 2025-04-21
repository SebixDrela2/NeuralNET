using NeutralNET.Matrices;

namespace NeutralNET.Framework;

public interface IModelRunner
{
    Matrix Input { get; init; }
    Func<Matrix> Forward { get; init; }
}
public class ModelRunner : IModelRunner
{
    public required Matrix Input { get; init; }
    public required Func<Matrix> Forward { get; init; }
}
