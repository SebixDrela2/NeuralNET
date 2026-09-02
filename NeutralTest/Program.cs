using NeutralNET.Activation;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Connected.Optimizers;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Neural;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Matrices;
using NeutralNET.Models;

namespace NeutralTest;

internal class Program
{
    private const int BatchSize = 64;

    static void Main()
    {
        RunCnnCifar10HundredImagesFast();
    }

    public static void RunCnnCifar10HundredImagesFast()
    {
        string dataDir = @"D:\Cifar\datasets\cifar-10-batches-bin";

        // 1. Load data
        var (trainImages, trainLabels, actualLabels) = Cifar10DataLoader.LoadBatches(
            dataDir: dataDir,
            batchSize: 10,
            maxSamples: 10000
        );

        var cnnConfig = new CnnArchitectureConfig
        {
            ConvLayers = new List<CnnLayerConfig>
        {
            new() { KernelHeight = 3, KernelWidth = 3, Filters = 16, Stride = 1, Padding = 1,
                    Activation = ActivationType.LeakyReLU, UseMaxPool = true, PoolSize = 2 },
            new() { KernelHeight = 3, KernelWidth = 3, Filters = 32, Stride = 1, Padding = 1,
                    Activation = ActivationType.LeakyReLU, UseMaxPool = true, PoolSize = 2 }
        },
            DenseArchitecture = [512, 256, 10],
            DenseHiddenActivation = ActivationType.LeakyReLU,
            OutputActivation = ActivationType.Softmax,
            // Use Adam optimizer with proper hyperparameters
            OptimizerConfig = new CnnOptimizerConfig
            {
                OptimizerType = CnnOptimizerType.Adam,
                LearningRate = 0.001f,
                WeightDecay = 0.0005f,
                Beta1 = 0.9f,
                Beta2 = 0.999f,
                Epsilon = 1e-8f
            }
        };

        var denseConfig = new NeuralNetworkConfig
        {
            LearningRate = 0.001f,      // Adam uses lower LR
            WeightDecay = 0.0005f,
            BatchSize = 10,
            Epochs = 1,
            DropoutRate = 0.0f,
            WithShuffle = true,
            OptimizerType = OptimizerType.Adam,
            Model = null
        };

        using var network = new CnnBuilder<Architecture>()
            .WithCnnConfig(cnnConfig)
            .WithDenseConfig(denseConfig)
            .WithInputSize(32, 32, 3)
            .Build();

        var validator = new CnnValidator();

        float learningRate = 0.001f;    // Lower LR for Adam
        int maxEpochs = 2000;
        int earlyStopPatience = 100;    // Stop if no improvement for 100 epochs
        float targetAccuracy = 0.99f;   // Stop when accuracy reaches 95%

        Console.WriteLine($"Training on {trainImages.Count} batches...");
        Console.WriteLine("Epoch\tLoss\tAccuracy\tBest");

        float bestAccuracy = 0f;
        int epochsSinceBest = 0;

        for (int epoch = 0; epoch < maxEpochs; epoch++)
        {
            float totalLoss = 0;
            for (int batchIdx = 0; batchIdx < trainImages.Count; batchIdx++)
            {
                float loss = network.TrainBatch(trainImages[batchIdx], trainLabels[batchIdx], learningRate);
                totalLoss += loss;
            }
            float avgLoss = totalLoss / trainImages.Count;

            // Evaluate every 10 epochs to save time
            if (true)
            {
                var result = validator.Validate(network, trainImages, trainLabels);
                float accuracy = result.Accuracy;

                // Track best accuracy for early stopping
                if (accuracy > bestAccuracy)
                {
                    bestAccuracy = accuracy;
                    epochsSinceBest = 0;
                    Console.WriteLine($"{epoch + 1,4}\t{avgLoss:F6}\t{accuracy:P2}\t*** NEW BEST ***");
                }
                else
                {
                    epochsSinceBest++;
                    Console.WriteLine($"{epoch + 1,4}\t{avgLoss:F6}\t{accuracy:P2}\t{bestAccuracy:P2}");
                }

                // Early stopping conditions
                if (accuracy >= targetAccuracy)
                {
                    Console.WriteLine($"\n🎯 Target accuracy {targetAccuracy:P2} reached! Stopping early at epoch {epoch + 1}");
                    break;
                }

                if (epochsSinceBest >= earlyStopPatience && bestAccuracy > 0.5f)
                {
                    Console.WriteLine($"\n⏹️ No improvement for {earlyStopPatience} epochs. Stopping early at epoch {epoch + 1}");
                    Console.WriteLine($"Best accuracy: {bestAccuracy:P2}");
                    break;
                }
            }
        }

        // Final evaluation with detailed results
        Console.WriteLine("\n=== FINAL EVALUATION ===");
        var finalResult = validator.Validate(network, trainImages, trainLabels);
        validator.PrintResults(finalResult);

        // Cleanup
        foreach (var img in trainImages) img.Dispose();
        foreach (var lbl in trainLabels) lbl.Dispose();
    }

    public static void RunNetwork()
    {
        var model = new SumBitsModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder<Architecture>(model)
            .WithArchitecture([32, 32])
            .WithEpochs(10000)
            .WithHiddenLayerActivation(ActivationType.ReLU)
            .WithOutputLayerActivation(ActivationType.Sigmoid)
            .WithBatchSize(BatchSize)
            .WithLearningRate(0.01f)
            .WithWeightDecay(1e-5f)
            .WithShuffle(true)
            .Build();

        var forward = network.RunModel();
        model.Validate(forward);
    }

    public static void RunNetworkDigit()
    {
        var model = new DigitModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder<Architecture>(model)
            .WithArchitecture([128, 64, 32])
            .WithEpochs(5000)
            .WithBatchSize(64)
            .WithHiddenLayerActivation(ActivationType.ReLU)
            .WithOutputLayerActivation(ActivationType.Sigmoid)
            .WithOptimizer(OptimizerType.AdamW)
            .WithLearningRate(0.001f)
            .WithWeightDecay(1e-4f)
            .WithBeta1(0.9f)
            .WithBeta2(0.999f)
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .Build();

        var forward = network.RunModel();
        model.Validate(forward);
    }

    public static void RunSumBitsModel()
    {
        var model = new SumBitsModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder<Architecture>(model)
            .WithArchitecture([64, 64, 64, 64])
            .WithEpochs(200)
            .WithHiddenLayerActivation(ActivationType.ReLU)
            .WithOutputLayerActivation(ActivationType.Sigmoid)
            .WithBatchSize(BatchSize)
            .WithBeta1(0.9f)
            .WithBeta2(0.999f)
            .WithLearningRate(1e-2f)
            .WithOptimizer(OptimizerType.SGD)
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .Build();

        var forward = network.RunModel();
        model.Validate(forward);
    }

    public static void RunSingleDigitTransformation()
    {
        var model = new BitMapTransformationModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder<Architecture>(model)
            .WithArchitecture([2, 2])
            .WithEpochs(100000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(5e-5f)
            .WithWeightDecay(1e-5f)
            .WithBeta1(0.9f)
            .WithBeta2(0.999f)
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .Build();

        var forward = network.RunModel();
    }

    public static void RunInfiniteFunction()
    {
        var model = new FunctionModel();
        var network = new NeuralNetworkBuilder<Architecture>(model)
            .WithArchitecture([32, 32])
            .WithEpochs(20000)
            .WithHiddenLayerActivation(ActivationType.ReLU)
            .WithOutputLayerActivation(ActivationType.Identity)
            .WithBatchSize(BatchSize)
            .WithBeta1(0.9f)
            .WithBeta2(0.999f)
            .WithLearningRate(1e-5f)
            .WithOptimizer(OptimizerType.Adam)
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .Build();

        var forward = network.RunDynamicModel();
        model.Validate(forward, network.Architecture);
    }
}

