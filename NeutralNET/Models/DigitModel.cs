using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Matrices;
using NeutralNET.Models;
using NeutralNET.Stuff;
using NeutralNET.Validators;
using System;

public class DigitModel : IModel, IValidator
{
    private const int VariantFontCount = 10;
    public const int PixelCount = GraphicsUtils.Width * GraphicsUtils.Height;
    public const int DigitLimit = 10;
    public const int NumClasses = 10;  // 10 digits (0-9)

    public NeuralMatrix TrainingInput { get; set; }
    public NeuralMatrix TrainingOutput { get; set; }

    private static readonly string[] _fontNames =
        ["Arial", "Times New Roman", "Georgia", "Verdana", "Tahoma"];

    private readonly int _rowCount;

    public DigitModel()
    {
        _rowCount = _fontNames.Length * DigitLimit * VariantFontCount;

        TrainingInput = NeuralMatrix.GetOrCreate(_rowCount, PixelCount);
        TrainingOutput = NeuralMatrix.GetOrCreate(_rowCount, NumClasses);  // 10 output neurons
    }

    public void Prepare()
    {
        var index = 0;

        for (var k = 0; k < VariantFontCount; k++)
        {
            for (var j = 0; j < _fontNames.Length; j++)
            {
                var pixelStructs = GraphicsUtils.GetDigitsDataSet(_fontNames[j]);

                for (var digit = 0; digit < DigitLimit; ++digit, ++index)
                {
                    // Input: Copy pixel values
                    var inputRow = TrainingInput.GetRowSpan(index);
                    var pixelStruct = pixelStructs[digit];
                    pixelStruct.Values.CopyTo(inputRow);

                    // Output: One-hot encoding
                    var outputRow = TrainingOutput.GetRowSpan(index);
                    outputRow.Clear();              // Reset all to 0
                    outputRow[digit] = 1.0f;        // Set the correct class to 1
                }
            }
        }
    }

    public void Validate(NeuralForward forward)
    {
        // Test on Arial without transformations for consistent results
        var pixelStructs = GraphicsUtils.GetDigitsDataSet("Arial", applyTransformation: false);
        var inputRow = TrainingInput.GetRowSpan(0);

        Console.WriteLine();
        Console.WriteLine("=== RAW OUTPUT VALUES (All 10 Neurons) ===");
        Console.WriteLine("Digit | Expected | [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    | Result");
        Console.WriteLine("------|----------|-----------------------------------------------------------------|--------");

        int correct = 0;

        for (var digit = 0; digit < DigitLimit; ++digit)
        {
            var pixelStruct = pixelStructs[digit];
            pixelStruct.Values.CopyTo(inputRow);

            var output = forward();
            var outputSpan = output.GetRowSpan(0);
            int predicted = GetPredictedDigit(output);
            float confidence = outputSpan[predicted];

            Console.Write($"  {digit}   |    {digit}     | ");

            for (int j = 0; j < NumClasses; j++)
            {
                Console.Write($"{outputSpan[j]:F4} ");
            }

            bool isCorrect = predicted == digit;
            if (isCorrect) correct++;

            Console.Write($"| Pred: {predicted} ({confidence:F4})");
            Console.WriteLine(isCorrect ? " ✓" : " ✗");
        }

        Console.WriteLine($"\nAccuracy: {correct}/{DigitLimit} = {100f * correct / DigitLimit:F2}%");
        Console.WriteLine();
    }

    private int GetPredictedDigit(NeuralMatrix output)
    {
        var span = output.GetRowSpan(0);
        int maxIndex = 0;
        float maxValue = span[0];

        for (int i = 1; i < span.Length; i++)
        {
            if (span[i] > maxValue)
            {
                maxValue = span[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}
