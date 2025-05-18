using NeutralNET.Matrices;
using NeutralNET.Stuff;

namespace NeutralNET.Models;

public class BitMapTransformationModel : IModel
{
    public const int InputSize = 16 * 16 * 3;
    public const int RgbPixelCount = GraphicsUtils.Height * GraphicsUtils.Width * 3;
    public const int GrayscalePixelCount = GraphicsUtils.Height * GraphicsUtils.Width;
    public NeuralMatrix TrainingInput { get ; set ; }
    public NeuralMatrix TrainingOutput { get ; set ; }

    public uint[] TrainingOutputStrideMask { get; }

    public BitMapTransformationModel()
    {
        TrainingInput = new NeuralMatrix(1, InputSize);
        TrainingOutput = new NeuralMatrix(1, RgbPixelCount);
        TrainingOutputStrideMask = TrainingOutput.StrideMasks;
    }

    public void Prepare()
    {
        var outputImage = GraphicsUtils.LoadImage(@"C:\Users\Seba\Documents\Desktop\Central-Nic-Http-master\eagle256.png");

        var random = new Random();
        var inputPixels = new float[InputSize];

        for (int i = 0; i < inputPixels.Length; i++)
        {
            inputPixels[i] = (float)random.NextDouble();
        }
        var outputPixels = GraphicsUtils.ImageToFloatRGB(outputImage);

        inputPixels.CopyTo(TrainingInput.GetRowSpan(0));
        outputPixels.CopyTo(TrainingOutput.GetRowSpan(0));
    }
}
