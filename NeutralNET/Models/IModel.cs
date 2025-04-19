using NeutralNET.Matrices;

namespace NeutralNET.Models;

public interface IModel
{
    public Matrix TrainingInput { get; init; }
    public Matrix TrainingOutput { get; init; }
}
