using NeutralNET.Framework;
using NeutralNET.Framework.Neural;
using NeutralNET.Models;
using NeutralNET.Stuff;

namespace NeutralTest;

internal class Program
{
    private const int BatchSize = 120;

    static void Main(string[] args)
    {
        RunNetworkDigit();
    }

    static void RunNetwork()
    {
        var bits = BitModelUtils.Bits;
        var model = new SumBitsModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder<Architecture>()
            .WithArchitecture(
                inputSize: bits * 2, 
                hiddenLayers: [32, 32], 
                outputSize: bits + 1)
            .WithEpochs(3000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(0.01f)
            .WithWeightDecay(1e-5f)
            .WithShuffle(true)
            .WithModel(model)
            .Build();

        var forward = network.Run();
        model.Validate(forward);
    }

    static void RunNetworkDigit()
    {
        var model = new DigitModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder<Architecture>()
            .WithArchitecture(
                inputSize: DigitModel.PixelCount,
                hiddenLayers: [64, 64, 64, 64, 64],
                outputSize: 1)
            .WithEpochs(2000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(3e-4f)
            .WithWeightDecay(1e-5f)
            .WithBeta1(0.9f)   
            .WithBeta2(0.999f) 
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .WithModel(model)
            .Build();

        var forward = network.Run();
        model.Validate(forward);
    }
}
