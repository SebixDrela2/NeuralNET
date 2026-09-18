using NeutralNET.Activation;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Connected.Optimizers;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Matrices;
using NeutralNET.Test.Data;

namespace NeutralTest;

internal class Program
{
    private const int BatchSize = 32;

    static void Main() => RunCnnNetwork();
    public static void RunDuniel()
    {
        var loader = DataLoaderFactory.Create(DataSourceType.Letters);

        var dataSet = loader.LoadCompleteDataset(
           batchSize: BatchSize,
           maxTrainSamples: 3000,
           maxTestSamples: 300
        );

        char[] chars = [
            ..EnumerateChars('A').Take(loader.NumClasses)
        ];
        CnnDisplayWriter displayWriter = new(chars, dataSet.TrainImages.Count);
        displayWriter.Clear();

        Span<float> buff = stackalloc float[(1 + loader.NumClasses) * BatchSize];
        float accuracyVal = 0;

        while (true)
        {
            int offset = 0;
            for (int y = 0; y < BatchSize; ++y)
            {
                var expected = Random.Shared.Next(loader.NumClasses);
                buff[offset++] = expected;

                float acc = 1;

                var prob = buff[offset..(offset += loader.NumClasses)];
                for (int x = 0; x < loader.NumClasses; ++x)
                {
                    float v = Random.Shared.NextSingle() * acc;
                    prob[x] = v;
                    acc -= v;
                }
                Random.Shared.Shuffle(prob);
            }

            var rng = Random.Shared.NextSingle() * Random.Shared.NextSingle() * Random.Shared.NextSingle() * Random.Shared.NextSingle();
            accuracyVal += rng * (1 - accuracyVal);
            displayWriter.Accuracy = float.Round(loader.NumClasses * accuracyVal) / loader.NumClasses;
            displayWriter.TotalLoss = dataSet.TrainImages.Count * Random.Shared.NextSingle();
            displayWriter.Update(buff);
            Thread.Sleep(250);
        }
    }

    public static void RunCnnNetwork()
    {
        // 1. Data Loading
        var datasetKey = DataSourceType.Letters;
        var loader = DataLoaderFactory.Create(datasetKey);
        var config = CnnTrainingConfig.CreateDefault(loader.NumClasses);

        config.DatasetKey = datasetKey;

        var dataSet = loader.LoadCompleteDataset(
            batchSize: config.BatchSize,
            maxTrainSamples: config.MaxTrainSamples,
            maxTestSamples: config.MaxTestSamples
        );

        // 3. Network Initialization
        var network = new CnnBuilder()
            .WithCnnConfig(config.CnnArchitecture)
            .WithDenseConfig(config.DenseConfig)
            .WithInputSize(config.BatchSize, 3, loader.ImageScale, loader.ImageScale)
            .Build();

        var validator = new CnnValidator();
        var trainer = new CnnTrainer(network, validator, config, loader);
        trainer.Train(dataSet, loader.NumClasses);

        Console.WriteLine("\n=== FINAL EVALUATION ===");
        var finalResult = validator.Validate(network, dataSet.TestImages, dataSet.TestLabels);
        validator.PrintResults(finalResult);

        CleanupDataset(dataSet);
    }

    private static void CleanupDataset(NeuralDataset dataSet)
    {
        foreach (var img in dataSet.TrainImages) img?.Dispose();
        foreach (var lbl in dataSet.TrainLabels) lbl?.Dispose();
        foreach (var img in dataSet.TestImages) img?.Dispose();
        foreach (var lbl in dataSet.TestLabels) lbl?.Dispose();
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
}
