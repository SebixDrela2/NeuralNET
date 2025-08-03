using NeutralNET.Matrices;
using NeutralNET.Stuff;

namespace NeutralNET.Models;

public class BitMapTransformationModel : IModel
{
    public const int InputSize = 16 * 16 * 3;
    public const int RgbPixelCount = GraphicsUtils.Height * GraphicsUtils.Width * 3;
    public const int GrayscalePixelCount = GraphicsUtils.Height * GraphicsUtils.Width;

    private readonly string[] Paths = ["greenEagle256.png", "blueWolf256.png"];

    public NeuralMatrix TrainingInput { get ; set ; }
    public NeuralMatrix TrainingOutput { get ; set ; }

    public BitMapTransformationModel()
    {
        TrainingInput = new NeuralMatrix(Paths.Length, RgbPixelCount);
        TrainingOutput = new NeuralMatrix(Paths.Length, RgbPixelCount);
    }

    public void Prepare()
    {
        var random = new Random();
        var row = 0;

        foreach (var (firstPath, secondPath) in Paths.Zip(Paths.Reverse()))
        {
            var outputImage = GraphicsUtils.LoadImage(secondPath);
            var inputImage = GraphicsUtils.LoadImage(firstPath);

            //var inputPixels = new float[InputSize];

            //for (int i = 0; i < inputPixels.Length; i++)
            //{
            //    inputPixels[i] = (float)random.NextDouble();
            //}

            var outputPixels = GraphicsUtils.ImageToFloatRGB(outputImage);
            var inputPixels = GraphicsUtils.ImageToFloatRGB(inputImage);

            inputPixels.CopyTo(TrainingInput.GetRowSpan(row));
            outputPixels.CopyTo(TrainingOutput.GetRowSpan(row));

            ++row;
        }
    }
}
