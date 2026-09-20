using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Test.Data;
using NeutralTest;

namespace NeutralNET.ImageEpochViewer;

internal static class Program
{
    static void Main()
    {
        ApplicationConfiguration.Initialize();

        var datasetKey = DataSourceType.Letters;
        var loader = DataLoaderFactory.Create(datasetKey);
        var config = CnnTrainingConfig.CreateDefault(loader.NumClasses);
        config.DatasetKey = datasetKey;

        var network = new CnnBuilder()
            .WithCnnConfig(config.CnnArchitecture)
            .WithDenseConfig(config.DenseConfig)
            .WithInputSize(config.BatchSize, 3, loader.ImageScale, loader.ImageScale)
            .Build();

        try
        {
            network.LoadData(config.DatasetKey, config.CheckpointDir);
            Console.WriteLine($"[INFO] Successfully loaded existing weights for {config.DatasetKey}.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARNING] Could not load weights for {config.DatasetKey}: {ex.Message}. Running with untrained weights.");
        }

        var mainForm = new LetterWindow(network);
        Application.Run(mainForm);
    }
}
