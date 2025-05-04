using NeutralNET.Framework;
using NeutralNET.Models;
using NeutralNET.Stuff;

namespace NeutralTest;

internal class Program
{
    private const int BatchSize = 64;
    static void Main(string[] args)
    {
        Visualize();
        RunNetwork();
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
            .WithArchitecture(bits * 2, 32, 32, bits + 1)
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
}
