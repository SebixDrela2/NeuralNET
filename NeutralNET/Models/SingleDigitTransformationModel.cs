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
        var arial = GraphicsUtils.GetDigitsDataSet("Arial", false);
        var comicsans = GraphicsUtils.GetDigitsDataSet("Comic Sans MS", false);

        var threeStruct = arial[5];
        var eightStruct = comicsans[3];

        var inputRow = TrainingInput.GetRowSpan(0);
        var outputRow = TrainingOutput.GetRowSpan(0);

        threeStruct.Values.CopyTo(inputRow);
        eightStruct.Values.CopyTo(outputRow);
    }
}
