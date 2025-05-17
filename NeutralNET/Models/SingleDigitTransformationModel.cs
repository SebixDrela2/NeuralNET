using NeutralNET.Matrices;
using NeutralNET.Stuff;

namespace NeutralNET.Models;

public class SingleDigitTransformationModel : IModel
{
    public const int PixelCount = 16 * 16;
    public NeuralMatrix TrainingInput { get ; set ; }
    public NeuralMatrix TrainingOutput { get ; set ; }

    public uint[] TrainingOutputStrideMask { get; }

    public SingleDigitTransformationModel()
    {
        TrainingInput = new NeuralMatrix(1, PixelCount);
        TrainingOutput = new NeuralMatrix(1, PixelCount);
        TrainingOutputStrideMask = TrainingOutput.StrideMasks;
    }

    public void Prepare()
    {
        var brightStructs = GraphicsUtils.GetDigitsDataSet("Arial", false);

        var threeStruct = brightStructs[3];
        var eightStruct = brightStructs[8];

        var inputRow = TrainingInput.GetRowSpan(0);
        var outputRow = TrainingOutput.GetRowSpan(0);

        threeStruct.Values.CopyTo(inputRow);
        eightStruct.Values.CopyTo(outputRow);
    }
}
