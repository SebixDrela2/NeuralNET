using NeutralNET.Activation;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Connected.Optimizers;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Framework.Neural.GPT;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Test.Data;

namespace NeutralTest;

internal class Program
{
    static void Main() => RunGPT();

    public static void RunGPT()
    {
        var buildFolder = Path.Combine(GlobalScope.BuildDirectory, "magnificency");
        var corpusFile = Path.Combine(buildFolder, "ShakespeareWork.txt");
        var checkpointFile = Path.Combine(buildFolder, "shakespeare_gpt.bin");

        var loader = new GPTTextDataLoader();
        loader.LoadCorpus(corpusFile);

        var config = new GptConfig
        {
            VocabSize = loader.VocabSize,
            ContextSize = 64,
            EmbedDim = 128,
            IntermediateDim = 256,
            NumHeads = 4,
            NumLayers = 2,
            LearningRate = 0.005f,
            MaxBatchSize = 64
        };

        using var gpt = new GptNeuralFramework(config);
        var batches = loader.GetBatches(config.ContextSize);

        Console.WriteLine($"[DataLoader] Created {batches.Count:N0} training batches.");
        var totalEpochs = 100;

        Console.WriteLine("\n--- Starting Training ---");

        for (int epoch = 1; epoch <= totalEpochs; epoch++)
        {
            Console.WriteLine($"\n--- Epoch {epoch:D2}/{totalEpochs:D2} ---");
            int batchIndex = 0;
            int totalBatches = batches.Count;
            var epochTimer = System.Diagnostics.Stopwatch.StartNew();

            foreach (var batch in batches)
            {
                gpt.TrainStep(batch);
                batchIndex++;

                if (batchIndex % 10 == 0 || batchIndex == totalBatches)
                {
                    float progress = (float)batchIndex / totalBatches * 100f;
                    double elapsedSec = epochTimer.Elapsed.TotalSeconds;
                    double itemsPerSec = batchIndex / elapsedSec;

                    Console.Write($"\rBatch {batchIndex}/{totalBatches} [{progress:F1}%] - {itemsPerSec:F1} batch/s");
                }
            }

            Console.WriteLine();
            var (avgLoss, accuracy) = GptTrainingRunner.Evaluate(gpt, batches);
            Console.WriteLine($"Epoch {epoch:D2} Finished | Loss: {avgLoss:F4} | Accuracy: {accuracy:F2}% | Time: {epochTimer.Elapsed.TotalSeconds:F2}s");
        }

        GptTrainingRunner.SaveModel(gpt, checkpointFile);
        GptTrainingRunner.LoadModel(gpt, checkpointFile);

        Console.WriteLine("\n--- Generating Text ---");
        var prompt = "To be or not to be";
        var promptTokens = loader.Encode(prompt).ToArray();

        var outputTokens = gpt.Generate(promptTokens, maxNewTokens: 150, temperature: 0.8f, topP: 0.9f);
        var generatedText = loader.Decode(outputTokens);

        Console.WriteLine($"Prompt: {prompt}");
        Console.WriteLine($"Generated Output:\n{generatedText}");
    }

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
