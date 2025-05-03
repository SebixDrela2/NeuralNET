using NeutralNET.Matrices;

namespace NeutralNET.Models;

public interface IModel
{
    public Matrix TrainingInput { get; set; }
    public Matrix TrainingOutput { get; set; }

    public uint[] TrainingOutputStrideMask { get; }
}
