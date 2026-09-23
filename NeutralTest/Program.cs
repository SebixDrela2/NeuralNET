using NeutralNET.Activation;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Connected.Optimizers;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Test.Data;

namespace NeutralTest;

internal class Program
{
    static void Main() => RunCnnNetwork();

    public static void RunCnnNetwork()
    {
        if (!GraphicsUtils.IsSupported) throw new NotSupportedException();

        // 1. Data Loading
        using var loader = DataLoaderFactory.Create<LetterDataLoader>();
        var config = CnnTrainingConfig.CreateDefault(loader.NumClasses);
        config.DatasetKey = LetterDataLoader.DataSourceType;

        using var dataSet = loader.LoadCompleteDataset(
            batchSize: config.BatchSize,
            maxTrainSamples: config.MaxTrainSamples,
            maxTestSamples: config.MaxTestSamples
        );

        // 3. Network Initialization
        var network = new CnnBuilder()
            .WithCnnConfig(config.CnnArchitecture)
            .WithDenseConfig(config.DenseConfig)
            .WithInputSize(config.BatchSize, 3, loader.ImageHeight, loader.ImageWidth)
            .Build();

        var validator = new CnnValidator();
        using var trainer = new CnnTrainer(network, validator, config, loader);
        trainer.Train(dataSet, loader.NumClasses);

        Console.WriteLine("\e[K");
        Console.WriteLine("=== FINAL EVALUATION ===\e[K");
        var finalResult = validator.Validate(network, dataSet.Test.ImagesData, dataSet.Train.LabelsData);
        validator.PrintResults(finalResult);
    }
}
