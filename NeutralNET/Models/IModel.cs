using NeutralNET.Matrices;

namespace NeutralNET.Models;

public interface IModel
{
    Matrix TrainingInput { get; set; }
    Matrix TrainingOutput { get; set; }
    Func<Matrix> Forward { get; set; }
    void Prepare();
    uint[] TrainingOutputStrideMask { get; }
}
