using NeutralNET.Matrices;
using NeutralNET.Stuff;

namespace NeutralNET.Models;

public class GrayScaleImageModel : IModel
{
    private const int InputSize = 2;
    private const int OutputSize = 1;

    private const int PixelCount = GraphicsUtils.PixelCount;

    public NeuralMatrix TrainingInput { get; set; }
    public NeuralMatrix TrainingOutput { get; set; }

    public uint[] TrainingOutputStrideMask { get; }

    public GrayScaleImageModel()
    {
        TrainingInput = new NeuralMatrix(PixelCount, InputSize);
        TrainingOutput = new NeuralMatrix(PixelCount, OutputSize);
        TrainingOutputStrideMask = TrainingOutput.StrideMasks;
    }

    public void Prepare()
    {
        var three = GraphicsUtils.GenerateCharPixelStruct('3', "Arial");
        var index = 0;

        float width = GraphicsUtils.Width - 1;
        float height = GraphicsUtils.Height - 1;

        for (var x = 0; x < GraphicsUtils.Width; x++)
        {
            for (var y = 0; y < GraphicsUtils.Height; y++, index++)
            {
                var inputRow = TrainingInput.GetRowSpan(index);
                var outputRow = TrainingOutput.GetRowSpan(index);

                float[] coordinatesArray = [x / width, y / height];
                var output = three.Values[index];

                coordinatesArray.CopyTo(inputRow);
                new float[] { output }.CopyTo(outputRow);
            }
        }
    }
}
