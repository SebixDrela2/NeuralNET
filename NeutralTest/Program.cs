using NeutralNET.Framework;
using NeutralNET.Framework.Neural;
using NeutralNET.Models;
using NeutralNET.Stuff;

namespace NeutralTest;

internal class Program
{
    private const int BatchSize = 64;

    static void Main(string[] args)
    {
        RunNetworkDigit();
    }

    public static void RunNetwork()
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

    public static void RunNetworkDigit()
    {
        var model = new DigitModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder<Architecture>()
            .WithArchitecture(
                inputSize: DigitModel.PixelCount,
                hiddenLayers: [64, 64, 64, 64, 64],
                outputSize: 1)
            .WithEpochs(3000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(1e-4f)
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

    public static void RunSingleDigitTransformation()
    {
        var model = new BitMapTransformationModel();
        model.Prepare();

        var network = new NeuralNetworkBuilder<Architecture>()
            .WithArchitecture(
                inputSize: BitMapTransformationModel.PixelCount,
                hiddenLayers: [2, 2],
                outputSize: BitMapTransformationModel.PixelCount)
            .WithEpochs(100000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(5e-5f)
            .WithWeightDecay(1e-5f)
            .WithBeta1(0.9f)
            .WithBeta2(0.999f)
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .WithModel(model)
            .Build();

        //var matrixes = network.RunEpoch();

        //foreach (var matrix in matrixes)
        //{
        //    matrix.Print("EPOCH");
        //    model.TrainingOutput.Print("OUTPUT");

        //    Thread.Sleep(1000);
        //}

        var forward = network.Run();
    }
}
