using NeutralNET.Activation;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Connected.Optimizers;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Test.Data;

namespace NeutralTest;

public class CnnTrainingConfig
{
    public const bool EnableGpu = CnnNeuralFramework.EnableGpu;
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
        const int BatchSize = 128;

        return new CnnTrainingConfig
        {
            CnnArchitecture = new CnnArchitectureConfig
            {
                ConvLayers =
                [
                    new() {
                        KernelHeight = 3, KernelWidth = 3, Filters = 32, Stride = 1, Padding = 1,
                        Activation = ActivationType.LeakyReLU, UseMaxPool = true, PoolSize = 2,
                        UseBatchNorm = true, BatchNormMomentum = 0.1f, BatchNormEpsilon = 1e-5f,
                    },
                    new() {
                        KernelHeight = 3, KernelWidth = 3, Filters = 64, Stride = 1, Padding = 1,
                        Activation = ActivationType.LeakyReLU, UseMaxPool = true, PoolSize = 2,
                        UseBatchNorm = true, BatchNormMomentum = 0.1f, BatchNormEpsilon = 1e-5f,
                    },
                    new() {
                        KernelHeight = 3, KernelWidth = 3, Filters = 128, Stride = 1, Padding = 1,
                        Activation = ActivationType.LeakyReLU, UseMaxPool = true, PoolSize = 2,
                        UseBatchNorm = true, BatchNormMomentum = 0.1f, BatchNormEpsilon = 1e-5f,
                    }
                ],
                DenseArchitecture = [64, numClasses],
                DenseHiddenActivation = ActivationType.LeakyReLU,
                OutputActivation = ActivationType.Softmax,

                OptimizerConfig = new CnnOptimizerConfig
                {
                    OptimizerType = CnnOptimizerType.AdamW,
                    LearningRate = 1e-4f,
                    WeightDecay = 0f,
                    Beta1 = 0.9f,
                    Beta2 = 0.999f,
                    Epsilon = 1e-8f
                }
            },

            DenseConfig = new NeuralNetworkConfig
            {
                LearningRate = 1e-4f,
                WeightDecay = 0f,
                BatchSize = BatchSize,
                Epochs = 200,
                DropoutRate = 0.1f,
                WithShuffle = true,
                OptimizerType = OptimizerType.AdamW,
            },

            LearningRate = 1e-4f,
            MaxTrainSamples = BatchSize * 64,
            MaxTestSamples = BatchSize,
        };
    }
}
