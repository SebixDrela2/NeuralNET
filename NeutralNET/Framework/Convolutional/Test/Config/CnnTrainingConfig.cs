using NeutralNET.Activation;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Connected.Optimizers;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Test.Data;

namespace NeutralTest;

public class CnnTrainingConfig
{
    public DataSourceType DatasetKey { get; set; } = DataSourceType.Letters;
    public int BatchSize => DenseConfig.BatchSize;
    public required int MaxTrainSamples { get; set; }
    public required int MaxTestSamples { get; set; }

    public float LearningRate { get; set; }
    public float TargetAccuracy { get; set; } = 1f;
    public float TargetLoss { get; set; } = 0.0001f;
    public int EarlyStopPatience { get; set; } = 300;
    public string CheckpointDir { get; set; } = Path.Join(BuildDirectory, "checkpoints");

    public CnnArchitectureConfig CnnArchitecture { get; set; } = new();
    public NeuralNetworkConfig DenseConfig { get; set; } = new();

    public static CnnTrainingConfig CreateDefault(int numClasses)
    {
        return new CnnTrainingConfig
        {
            CnnArchitecture = new CnnArchitectureConfig
            {
                ConvLayers =
                [
                   // Layer 1: 64x64 -> 32x32 (8 filters)
                   new() {
                       KernelHeight = 3, KernelWidth = 3, Filters = 8, Stride = 1, Padding = 1,
                       Activation = ActivationType.LeakyReLU, UseMaxPool = true, PoolSize = 2
                   },
                   // Layer 2: 32x32 -> 16x16 (16 filters)
                   new() {
                       KernelHeight = 3, KernelWidth = 3, Filters = 16, Stride = 1, Padding = 1,
                       Activation = ActivationType.LeakyReLU, UseMaxPool = true, PoolSize = 2
                   },
                   // Layer 2: 16x16 -> 8x8 (32 filters)
                    new() {
                       KernelHeight = 3, KernelWidth = 3, Filters = 32, Stride = 1, Padding = 1,
                       Activation = ActivationType.LeakyReLU, UseMaxPool = true, PoolSize = 2
                   }
                ],
                // Wide single hidden layer avoids information loss on 26 output classes
                DenseArchitecture = [64, numClasses],
                DenseHiddenActivation = ActivationType.LeakyReLU,
                OutputActivation = ActivationType.Softmax,
                OptimizerConfig = new CnnOptimizerConfig
                {
                    OptimizerType = CnnOptimizerType.Adam,
                    LearningRate = 0.0005f,
                    WeightDecay = 1e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.999f,
                    Epsilon = 1e-8f
                }
            },
            DenseConfig = new NeuralNetworkConfig
            {
                LearningRate = 0.0005f,
                WeightDecay = 1e-4f,
                BatchSize = 1 << 8,
                Epochs = 100,
                DropoutRate = 0.1f,
                WithShuffle = true,
                OptimizerType = OptimizerType.Adam,
            },
            LearningRate = 0.0005f,
            MaxTrainSamples = 1 << 12,
            MaxTestSamples = 1 << 8,
        };
    }
}
