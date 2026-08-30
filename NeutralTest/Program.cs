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
            maxSamples: 100
        );

        var cnnConfig = new CnnArchitectureConfig
        {
            ConvLayers = new List<CnnLayerConfig>
        {
            new() { KernelHeight = 3, KernelWidth = 3, Filters = 16, Stride = 1, Padding = 1,
                    Activation = ActivationType.ReLU, UseMaxPool = true, PoolSize = 2 },
            new() { KernelHeight = 3, KernelWidth = 3, Filters = 32, Stride = 1, Padding = 1,
                    Activation = ActivationType.ReLU, UseMaxPool = true, PoolSize = 2 }
        },
            DenseArchitecture = new[] { 128, 10 },
            DenseHiddenActivation = ActivationType.ReLU,
            OutputActivation = ActivationType.Softmax
        };

        var denseConfig = new NeuralNetworkConfig
        {
            LearningRate = 0.005f,
            WeightDecay = 0.0001f,
            BatchSize = 10,
            Epochs = 1,
            DropoutRate = 0.0f,
            WithShuffle = true,
            OptimizerType = OptimizerType.SGD,
            Model = null
        };

        using var network = new CnnBuilder<Architecture>()
            .WithCnnConfig(cnnConfig)
            .WithDenseConfig(denseConfig)
            .WithInputSize(32, 32, 3)
            .Build();

        var validator = new CnnValidator();

        float learningRate = 0.005f;
        int epochs = 2000;

        Console.WriteLine($"Training on {trainImages.Count} batches...");
        Console.WriteLine("Epoch\tLoss\tAccuracy");

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            float totalLoss = 0;
            for (int batchIdx = 0; batchIdx < trainImages.Count; batchIdx++)
            {
                float loss = network.TrainBatch(trainImages[batchIdx], trainLabels[batchIdx], learningRate);
                totalLoss += loss;
            }
            float avgLoss = totalLoss / trainImages.Count;

            if (epoch % 25 == 0 || epoch == epochs - 1)
            {
                var result = validator.Validate(network, trainImages, trainLabels);
                Console.WriteLine($"{epoch + 1,4}\t{avgLoss:F6}\t{result.Accuracy:P2}");
            }
        }

        var finalResult = validator.Validate(network, trainImages, trainLabels);
        validator.PrintResults(finalResult);

        foreach (var img in trainImages) img.Dispose();
        foreach (var lbl in trainLabels) lbl.Dispose();
    }

    private static int GetLabelIndex(NeuralMatrix labelMatrix)
    {
        var row = labelMatrix.GetRowSpan(0);

        for (int i = 0; i < row.Length; i++)
        {
            if (row[i] > 0.5f)
            {
                return i;
            }
        }

        return -1;
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

