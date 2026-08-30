using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public class CnnValidator
{
    public ValidationResult Validate(CnnNetwork<Architecture> network, List<CnnMatrix> images, List<NeuralMatrix> labels, int maxSamplesToShow = int.MaxValue)
    {
        int correct = 0;
        int total = 0;
        var samplePredictions = new List<SamplePrediction>();

        for (int batchIdx = 0; batchIdx < images.Count; batchIdx++)
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
        Console.WriteLine($"\n=== VALIDATION RESULTS ===");
        Console.WriteLine($"Accuracy: {result.Accuracy:P2} ({result.Correct}/{result.Total})");

        Console.WriteLine($"\n=== FIRST {samplesToShow} SAMPLE PREDICTIONS ===");
        Console.WriteLine("Sample\tPred\tActual\tResult\tProbabilities (0-9)");
        Console.WriteLine("------\t----\t------\t------\t-----------------");

        foreach (var sample in result.SamplePredictions.Take(samplesToShow))
        {
            string status = sample.IsCorrect ? "✓" : "✗";
            string probs = string.Join(" ", sample.Probabilities.Select(p => p.ToString("F3")));

            Console.WriteLine($"{sample.SampleIndex,5}\t{sample.Predicted,3}\t{sample.Actual,5}\t{status,4}\t{probs}");
        }
    }

    private static int ArgMax(Span<float> row)
    {
        int maxIdx = 0;
        float maxVal = row[0];

        for (int i = 1; i < row.Length; i++)
        {
            if (row[i] > maxVal)
            {
                maxVal = row[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
