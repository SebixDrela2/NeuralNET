using NeutralNET.Framework;
using NeutralNET.Models;
using NeutralNET.Validators;

namespace NeutralTest;

internal class Program
{
    static void Main(string[] args)
    {
        const int Bits = 6;
        const int BatchSize = 64;
        var network = new NeuralNetworkBuilder()
            .WithArchitecture(Bits * 2, 32, 32, Bits * 2)
            .WithEpochs(3000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(0.01f)
            .WithWeightDecay(1e-5f)
            .WithModel(new SumBitsModel())
            .Build();

        var modelRunner = network.Run();
        var validator = new SumBitsModelValidator(modelRunner);
        validator.Validate();
    }
}
