using NeutralNET.Framework;
using NeutralNET.Models;
using NeutralNET.Validators;

namespace NeutralTest;

internal class Program
{
    static void Main(string[] args)
    {
        var network = new NeuralNetworkBuilder()
            .WithArchitecture(8, 12, 12, 8)
            .WithEpochs(200 * 100)
            .WithBatchSize(64)
            .WithLearningRate(0.01f)
            .WithWeightDecay(1e-5f)
            .WithModel(new BitModel())
            .Build();

        var modelRunner = network.Run();
        var validator = new BitModelValidator(modelRunner);
        validator.Validate();
    }
}
