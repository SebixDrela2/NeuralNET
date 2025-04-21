using NeutralNET.Framework;
using NeutralNET.Models;
using NeutralNET.Validators;

namespace NeutralTest;

internal class Program
{
    static void Main(string[] args)
    {
        var network = new NeuralNetworkBuilder()
            .WithArchitecture(12, 32, 32, 12)
            .WithEpochs(2000)
            .WithBatchSize(200)
            .WithLearningRate(0.01f)
            .WithWeightDecay(1e-5f)
            .WithModel(new SumBitsModel())
            .Build();

        var modelRunner = network.Run();
        var validator = new SumBitsModelValidator(modelRunner);
        validator.Validate();
    }
}
