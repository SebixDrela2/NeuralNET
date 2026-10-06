using System.Diagnostics;
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
            ContextSize = 256,
            EmbedDim = 128,
            IntermediateDim = 512,
            NumHeads = 4,
            NumLayers = 2,
            LearningRate = 0.0003f,
            MaxBatchSize = 64
        };

        using var gpt = new GptNeuralFramework(config);

        var sequenceChunks = loader.GetBatches(config.ContextSize);
        var miniBatches = sequenceChunks.Chunk(config.MaxBatchSize).ToList();

        Console.WriteLine($"[DataLoader] Created {miniBatches.Count:N0} mini-batches (Batch Size: {config.MaxBatchSize}).");

        var totalEpochs = 10;

        Console.WriteLine("\n--- Starting Training ---");

        const int warmupSteps = 20;
        int globalStep = 0;
        float baseLr = config.LearningRate;

        int batchesPerEpoch = miniBatches.Count;
        int totalSteps = batchesPerEpoch * totalEpochs;

        for (int epoch = 1; epoch <= totalEpochs; epoch++)
        {
            Console.WriteLine($"\n--- Epoch {epoch:D2}/{totalEpochs:D2} ---");

            var batchIndex = 0;
            var totalBatches = miniBatches.Count;
            var epochTimer = Stopwatch.StartNew();

            foreach (var miniBatch in miniBatches)
            {
                globalStep++;
                float warmupFactor = MathF.Min(1f, (float)globalStep / warmupSteps);
                float progress = MathF.Min(1f, (float)globalStep / totalSteps);
                float decayFactor = 0.5f * (1f + MathF.Cos(MathF.PI * progress));   // 1 -> 0
                float currentLr = baseLr * warmupFactor * decayFactor;
                gpt.SetLearningRate(currentLr);

                var batchTimer = Stopwatch.StartNew();
                gpt.TrainStep(miniBatch);
                batchTimer.Stop();
                batchIndex++;

                if (GptNeuralFramework.DiagnosticsEnabled)
                {
                    unsafe
                    {
                        float* q = gpt.Layers[0].Wq.Pointer;
                        float* out2 = gpt.OutputProjection.Pointer;
                        float* tok = gpt.TokenEmbeddings.Pointer;
                        float* pos = gpt.PositionalEmbeddings.Pointer;
                        float* logits = gpt.LogitsOutput.Pointer;

                        // Use the actual matrix dimensions and strides.
                        int qCount = gpt.Layers[0].Wq.Rows * gpt.Layers[0].Wq.ColumnsStride;
                        int outCount = gpt.OutputProjection.Rows * gpt.OutputProjection.ColumnsStride;
                        int tokCount = gpt.TokenEmbeddings.Rows * gpt.TokenEmbeddings.ColumnsStride;
                        int posCount = gpt.PositionalEmbeddings.Rows * gpt.PositionalEmbeddings.ColumnsStride;
                        int logCount = gpt.LogitsOutput.Rows * gpt.LogitsOutput.ColumnsStride;

                        float qMax = 0f, outMax = 0f, tokMax = 0f, posMax = 0f, logMax = 0f;
                        for (int i = 0; i < qCount; i++) if (MathF.Abs(q[i]) > qMax) qMax = MathF.Abs(q[i]);
                        for (int i = 0; i < outCount; i++) if (MathF.Abs(out2[i]) > outMax) outMax = MathF.Abs(out2[i]);
                        for (int i = 0; i < tokCount; i++) if (MathF.Abs(tok[i]) > tokMax) tokMax = MathF.Abs(tok[i]);
                        for (int i = 0; i < posCount; i++) if (MathF.Abs(pos[i]) > posMax) posMax = MathF.Abs(pos[i]);
                        for (int i = 0; i < logCount; i++) if (MathF.Abs(logits[i]) > logMax) logMax = MathF.Abs(logits[i]);

                        Console.WriteLine($"[batch-diag] |Wq|max={qMax:G4} |Wo|max={outMax:G4} |tok|max={tokMax:G4} |pos|max={posMax:G4} |logits|max={logMax:G4}");
                    }
                }

                if (batchIndex == 1 || (batchIndex & 15) == 0 || batchIndex == totalBatches)
                {
                    float progress2 = (float)batchIndex / totalBatches * 100f;
                    double batchMs = batchTimer.Elapsed.TotalMilliseconds;
                    double elapsedSec = epochTimer.Elapsed.TotalSeconds;
                    double itemsPerSec = batchIndex / elapsedSec;

                    Console.WriteLine($"Batch {batchIndex}/{totalBatches} [{progress2:F1}%] - Step Time: {batchMs:F2} ms | LR: {gpt.Config.LearningRate:E2} | Avg Speed: {itemsPerSec:F1} batch/s");
                }
            }

            var (avgLoss, accuracy) = GptTrainingRunner.Evaluate(gpt, miniBatches);
            Console.WriteLine($"Epoch {epoch:D2} Finished | Loss: {avgLoss:F4} | Accuracy: {accuracy:F2}% | Total Epoch Time: {epochTimer.Elapsed.TotalSeconds:F2}s");
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
