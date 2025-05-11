using NeutralNET.Framework;
using NeutralNET.Models;
using NeutralNET.Stuff;

namespace NeutralTest;

internal class Program
{
    private const int BatchSize = 10;

    static void Main(string[] args)
    {
        Visualize();
        RunNetworkDigit();
    }

    static void Visualize()
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
    }

    static void RunNetwork()
    {
        var bits = BitModelUtils.Bits;
        var model = new SumBitsModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder()
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

        network.Run();
        model.Validate();
    }

    static void RunNetworkDigit()
    {
        var model = new DigitModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder()
            .WithArchitecture(
                inputSize: DigitModel.PixelCount, 
                hiddenLayers: [128, 128, 64, 32, 32, 32, 32], 
                outputSize: 1)
            .WithEpochs(20000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(1e-3f)
            .WithWeightDecay(1e-5f)
            .WithShuffle(true)
            .WithModel(model)
            .Build();

        network.Run();
        model.Validate();
    }
}
