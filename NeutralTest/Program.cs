using NeutralNET.Activation;
using NeutralNET.Framework;
using NeutralNET.Framework.Neural;
using NeutralNET.Framework.Optimizers;
using NeutralNET.Models;

namespace NeutralTest;

internal class Program
{
    private const int BatchSize = 64;

    static void Main()
    {
        RunInfiniteFunction();
    }

    public static void RunNetwork()
    {
        var model = new SumBitsModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder<Architecture>(model)
            .WithArchitecture([32, 32])
            .WithEpochs(10000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(0.01f)
            .WithWeightDecay(1e-5f)
            .WithShuffle(true)
            .Build();

        var forward = network.RunModel();
        model.Validate(forward);
    }

    public static void RunNetworkDigit()
    {
        var model = new DigitModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder<Architecture>(model)
            .WithArchitecture([16, 16, 16])
            .WithEpochs(3000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(1e-10f)
            .WithWeightDecay(1e-5f)
            .WithBeta1(0.9f)   
            .WithBeta2(0.999f) 
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .Build();

        var forward = network.RunModel();
        model.Validate(forward);
    }

    public static void RunSumBitsModel()
    {
        var model = new SumBitsModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder<Architecture>(model)
            .WithArchitecture([16, 16, 16, 16])
            .WithEpochs(10000)
            .WithHiddenLayerActivation(ActivationType.ReLU)
            .WithOutputLayerActivation(ActivationType.Sigmoid)
            .WithBatchSize(BatchSize)
            .WithBeta1(0.9f)
            .WithBeta2(0.999f)
            .WithLearningRate(1e-2f)
            .WithOptimizer(OptimizerType.SGD)
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .Build();
        
        var forward = network.RunModel();
        model.Validate(forward);
    }

    public static void RunSingleDigitTransformation()
    {
        var model = new BitMapTransformationModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder<Architecture>(model)
            .WithArchitecture([2, 2])
            .WithEpochs(100000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(5e-5f)
            .WithWeightDecay(1e-5f)
            .WithBeta1(0.9f)
            .WithBeta2(0.999f)
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .Build();

        //var matrixes = network.RunEpoch();

        //foreach (var matrix in matrixes)
        //{
        //    matrix.Print("EPOCH");
        //    model.TrainingOutput.Print("OUTPUT");

        //    Thread.Sleep(1000);
        //}

        var forward = network.RunModel();
    }

    public static void RunInfiniteFunction()
    {
        var model = new FunctionModel();
        var network = new NeuralNetworkBuilder<Architecture>(model)
            .WithArchitecture([32, 32])
            .WithEpochs(5000)
            .WithHiddenLayerActivation(ActivationType.LeakyReLU)
            .WithOutputLayerActivation(ActivationType.Identity)
            .WithBatchSize(BatchSize)
            .WithBeta1(0.9f)
            .WithBeta2(0.999f)
            .WithLearningRate(1e-5f)
            .WithOptimizer(OptimizerType.Adam)
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .Build();

        var forward = network.RunDynamicModel();
        model.Validate(forward, network.Architecture);
    }
}

