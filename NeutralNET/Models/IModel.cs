using NeutralNET.Matrices;

namespace NeutralNET.Models;

public interface IModel
{
    NeuralMatrix TrainingInput { get; set; }
    NeuralMatrix TrainingOutput { get; set; }
    void Prepare();
    uint[] TrainingOutputStrideMask { get; }
}
