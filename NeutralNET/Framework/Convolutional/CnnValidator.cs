using System.Runtime.InteropServices;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public class CnnValidator
{
    public ValidationResult Validate(CnnNetwork network, ReadOnlySpan<CnnMatrix> images, ReadOnlySpan<NeuralMatrix> labels, int maxSamplesToShow = int.MaxValue)
    {
        int correct = 0;
        int total = 0;
        var samplePredictions = new List<SamplePrediction>();

        for (int batchIdx = 0; batchIdx < images.Length; batchIdx++)
        {
            using var output = network.Forward(images[batchIdx]);
            var label = labels[batchIdx];

            for (int i = 0; i < output.Rows; i++)
            {
                int pred = ArgMax(output.GetRowSpan(i));
                int actual = ArgMax(label.GetRowSpan(i));

                if (pred == actual)
                {
                    correct++;
                }

                if (samplePredictions.Count < maxSamplesToShow)
                {
                    var probs = output.GetRowSpan(i).ToArray();
                    samplePredictions.Add(new SamplePrediction
                    {
                        SampleIndex = total,
                        Predicted = pred,
                        Actual = actual,
                        IsCorrect = pred == actual,
                        Probabilities = probs
                    });
                }
                total++;
            }
        }

        return new ValidationResult
        {
            Accuracy = (float)correct / total,
            Correct = correct,
            Total = total,
            SamplePredictions = samplePredictions
        };
    }

    public void PrintResults(ValidationResult result, int samplesToShow = 100)
    {
        Console.WriteLine("\e[K");
        Console.WriteLine($"=== VALIDATION RESULTS ===\e[K");
        Console.WriteLine($"Accuracy: {result.Accuracy:P2} ({result.Correct}/{result.Total})\e[K");

        Console.WriteLine("\e[K");
        Console.WriteLine($"=== FIRST {samplesToShow} SAMPLE PREDICTIONS ===\e[K");
        Console.WriteLine($" Sample | Pred | Actual | Result | Probabilities \e[K");
        Console.WriteLine($"--------|------|--------|--------|---------------\e[K");

        foreach (var sample in result.SamplePredictions.Take(samplesToShow))
        {
            string status = sample.IsCorrect ? "✓" : "✗";
            string probs = string.Join(" ", sample.Probabilities.Select(p => $"{p:f3}"));

            Console.WriteLine($"{sample.SampleIndex,8}|{sample.Predicted,6}|{sample.Actual,8}|{status,8}|{probs}\e[K");
        }
    }

    private static int ArgMax(Span<float> row) => row.OffsetOf(in row.Max());
}
