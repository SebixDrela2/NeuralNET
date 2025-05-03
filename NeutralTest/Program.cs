using NeutralNET.Framework;
using NeutralNET.Models;
using NeutralNET.Stuff;
using NeutralNET.Validators;

namespace NeutralTest;

internal class Program
{
    static void Main(string[] args)
    {
        var bits = BitModelUtils.Bits;
        const int BatchSize = 64;
        var network = new NeuralNetworkBuilder()
            .WithArchitecture(bits * 2, 32, 32, bits * 2)
            .WithEpochs(3000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(0.01f)
            .WithWeightDecay(1e-5f)
            .WithShuffle(true)
            .WithModel(new SumBitsModel())
            .Build();

        var modelRunner = network.Run();
        var validator = new SumBitsModelValidator(modelRunner);
        validator.Validate();
    }
}
