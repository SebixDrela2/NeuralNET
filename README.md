## How to run/validate.

In order to properly run a neural network, you must specify it's config parameters.
Most importantly network architecture, batchsize, epochs (training count) and the learning rate.

**NeuralNetwork** example

```cs
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

```
