using System.Text;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Test.Data;

namespace NeutralTest;

public class CnnTrainer : IDisposable
{
    private readonly CnnNetwork _network;
    private readonly CnnValidator _validator;
    private readonly CnnTrainingConfig _config;
    private readonly DataLoaderBase _loader;

    public CnnTrainer(CnnNetwork network, CnnValidator validator, CnnTrainingConfig config, DataLoaderBase loader)
    {
        _network = network;
        _validator = validator;
        _config = config;
        _loader = loader;
    }

    public void Train(NeuralDataSet dataSet, int numClasses)
    {
        _network.LoadData(_config.DatasetKey, _config.CheckpointDir);
        char[] chars = [.. EnumerateChars('A').Take(numClasses)];

        var trainN = dataSet.Train.ImagesData.Length;

        var trainImg = dataSet.Train.ImagesData;
        var trainLbl = dataSet.Train.LabelsData;

        var display = new CnnDisplayWriter(chars, trainN);

        display.Clear();
        Span<float> results = new float[(1 + numClasses) * _config.BatchSize];
        var indexes = Enumerable.Range(0, trainN).ToArray();

        while (true)
        {
            Random.Shared.Shuffle(indexes);
            var totalLoss = 0.0f;

            //_network.SetPerf();

            for (var batchIdx = 0; batchIdx < trainN; batchIdx++)
            {
                var index = indexes[batchIdx];
                var loss = _network.TrainBatch(trainImg[index], trainLbl[index], _config.LearningRate);

                totalLoss += loss;
            }

            //_network.Report();

            if (!ProcessLoss(dataSet, numClasses, display, results, totalLoss))
            {
                break;
            }
        }
    }

    private bool ProcessLoss(NeuralDataSet dataSet, int numClasses, CnnDisplayWriter display, Span<float> results, float totalLoss)
    {
        var testImg = dataSet.Test.ImagesData;
        var testLbl = dataSet.Test.LabelsData;
        if (testImg is []) throw new InvalidOperationException();

        display.TotalLoss = totalLoss;

        var result = _validator.Validate(_network, testImg, testLbl);
        display.Accuracy = result.Accuracy;

        var offset = 0;
        var sampleBatch = testImg[0];
        var sampleLabels = testLbl[0];
        var pred = _network.Forward(sampleBatch);

        var numSamples = Math.Min(numClasses, sampleBatch.Batch);

        var orderedSamples = Enumerable
            .Range(0, sampleBatch.Batch)
            .Select(i =>
            {
                var actual = GetActualLabelFromRow(sampleLabels, i, numClasses);
                var pred2 = pred.GetRowSpan(i);
                //var maxError = 0f;
                var maxErr = (Index: 0, Value: 0f);
                var maxArg = (Index: 0, Value: 0f);

                for (var j = 0; j < pred2.Length; ++j)
                {
                    float arg = pred2[j];
                    float error = j == actual ? 1 - arg : arg;

                    if (j == 0)
                    {
                        maxArg = (j, arg);
                        maxErr = (j, error);
                    }
                    else
                    {
                        if (arg > maxArg.Value) maxArg = (j, arg);
                        if (error > maxErr.Value) maxErr = (j, error);
                    }
                }
                return (Index: i, Letter: actual, Result: maxArg.Index == actual, MaxError: maxErr.Value);
            })
            .OrderBy(x => x.Letter)
            .ThenByDescending(x => x.MaxError)
            .ToArray();
        var distinctOrderedSamples = orderedSamples.GroupBy(
            x => x.Letter,
            (k, x) => (Letter: k, Items: x.OrderByDescending(y => y.MaxError).Select(y => (y.Index, y.Result, y.MaxError)).ToArray())
        ).ToArray();

        for (int i = 0; i < distinctOrderedSamples.Length; i++)
        {
            var (letter, items) = distinctOrderedSamples[i];

            results[offset++] = letter;
            var probs = results[offset..(offset += numClasses)];
            var topIndex = items[0].Index;
            pred.GetRowSpan(topIndex).CopyTo(probs);
        }

        return DisplayWithControlFlow(display, dataSet, results[..offset], distinctOrderedSamples);
    }

    private bool DisplayWithControlFlow(CnnDisplayWriter display, NeuralDataSet dataset, Span<float> results, (int Letter, (int Index, bool Result, float MaxError)[] Items)[] results2)
    {
        display.Update(results, results2);

        if (display.EpochsSinceBest == 1)
        {
            _network.SaveData(_config.DatasetKey, _config.CheckpointDir);
            Console.WriteLine($"Saved Weights!");
        }

        if (display.AvgLoss <= _config.TargetLoss)
        {
            _network.SaveData(_config.DatasetKey, _config.CheckpointDir);
            Console.WriteLine($"\n🎯 Target accuracy {_config.TargetAccuracy:P2} reached! Stopping early at epoch {display.Epoch}");

            return false;
        }

        if (display.EpochsSinceBest >= _config.EarlyStopPatience)
        {
            Console.WriteLine($"\n⏹️ No improvement for {_config.EarlyStopPatience} epochs. Stopping early at epoch {display.Epoch}");
            Console.WriteLine($"Best accuracy: {display.BestAccuracy:P2}");
            _network.LoadData(_config.DatasetKey, _config.CheckpointDir);

            return false;
        }

        if (display.Accuracy is >= 1)
        {
            Console.Write("\e[J");
            _loader.ReloadCompleteDataset(dataset, _config);
            display.EpochsSinceBest = 0;
            display.BestAccuracy = 0;
            Console.WriteLine("Reloaded dataset.");
        }

        return true;
    }

    public static IEnumerable<char> CharRange(char first, char last)
    {
        if (first > last) (first, last) = (last, first);
        return EnumerateChars(first).TakeWhile(c => c <= last);
    }

    public static IEnumerable<char> EnumerateChars(char start)
    {
        for (char c = start; ; ++c)
        {
            yield return c;
            if (c is char.MaxValue) break;
        }
    }

    private static int GetActualLabelFromRow(NeuralMatrix labelMatrix, int row, int numClasses)
    {
        int maxIndex = -1;
        float maxValue = -1f;

        for (int i = 0; i < labelMatrix.UsedColumns; i++)
        {
            float val = labelMatrix.At(row, i);
            if (val > maxValue)
            {
                maxValue = val;
                maxIndex = i;
            }
        }

        if (maxIndex >= numClasses)
        {
            maxIndex = numClasses - 1;
        }

        return maxIndex;
    }

    public void Dispose()
    {
        _network.Dispose();
        _loader.Dispose();
    }
}
