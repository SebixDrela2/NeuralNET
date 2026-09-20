using NeutralNET.Activation;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Connected.Optimizers;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Test.Data;

namespace NeutralTest;

public class CnnTrainingConfig
{
    public DataSourceType DatasetKey { get; set; } = DataSourceType.Letters;
    public int MaxTrainSamples { get; set; } = 30000;
    public int MaxTestSamples { get; set; } = 1000;

    public int BatchSize { get; set; } = 1024;

    public float LearningRate { get; set; } = 0.0005f;
    public float TargetAccuracy { get; set; } = 1f;
    public float TargetLoss { get; set; } = 0.0001f;
    public int EarlyStopPatience { get; set; } = 300;
    public string CheckpointDir { get; set; } = @"C:\Users\Sebastian\source\repos\NeutralNET\NeutralTest\bin\Release\net10.0\checkpoints";

    public CnnArchitectureConfig CnnArchitecture { get; set; } = new();
    public NeuralNetworkConfig DenseConfig { get; set; } = new();

    public static CnnTrainingConfig CreateDefault(int numClasses = 26)
    {
        return new CnnTrainingConfig
        {
            CnnArchitecture = new CnnArchitectureConfig
            {
                ConvLayers =
                [
                   // Layer 1: 64x64 -> 32x32 (8 filters)
                   new() {
                       KernelHeight = 3, KernelWidth = 3, Filters = 16, Stride = 1, Padding = 1,
                       Activation = ActivationType.LeakyReLU, UseMaxPool = true, PoolSize = 2
                   },
                   // Layer 2: 32x32 -> 16x16 (16 filters)
                   new() {
                       KernelHeight = 3, KernelWidth = 3, Filters = 32, Stride = 1, Padding = 1,
                       Activation = ActivationType.LeakyReLU, UseMaxPool = true, PoolSize = 2
                   },
                   // Layer 2: 16x16 -> 8x8 (32 filters)
                    new() {
                       KernelHeight = 3, KernelWidth = 3, Filters = 64, Stride = 1, Padding = 1,
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
                    WeightDecay = 5e-4f,
                    Beta1 = 0.9f,
                    Beta2 = 0.999f,
                    Epsilon = 1e-8f
                }
            },
            DenseConfig = new NeuralNetworkConfig
            {
                LearningRate = 0.0005f,
                WeightDecay = 1e-4f,
                BatchSize = 1024,
                Epochs = 100,
                DropoutRate = 0.2f,
                WithShuffle = true,
                OptimizerType = OptimizerType.Adam,
                Model = null
            }
        };
    }
}
