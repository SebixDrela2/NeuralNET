using NeutralNET.Matrices;
using NeutralNET.Stuff;

namespace NeutralNET.Models;

public class BitMapTransformationModel : IModel
{
    public const int PixelCount = GraphicsUtils.Height * GraphicsUtils.Width;
    public NeuralMatrix TrainingInput { get ; set ; }
    public NeuralMatrix TrainingOutput { get ; set ; }

    public uint[] TrainingOutputStrideMask { get; }

    public BitMapTransformationModel()
    {
        TrainingInput = new NeuralMatrix(1, PixelCount);
        TrainingOutput = new NeuralMatrix(1, PixelCount);
        TrainingOutputStrideMask = TrainingOutput.StrideMasks;
    }

    public void Prepare()
    {
        var threeStruct = GraphicsUtils.GenerateCharPixelStruct('@', "Arial");
        var eightStruct = GraphicsUtils.GenerateCharPixelStruct('&', "Comic Sans MS");

        var inputRow = TrainingInput.GetRowSpan(0);
        var outputRow = TrainingOutput.GetRowSpan(0);

        threeStruct.Values.CopyTo(inputRow);
        eightStruct.Values.CopyTo(outputRow);
    }
}
