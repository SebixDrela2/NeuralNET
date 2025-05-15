using NeutralNET.Models;

namespace NeutralNET.Framework;

public class NeuralNetworkBuilder<TArch> where TArch : IArchitecture<TArch>
{
    private readonly NeuralNetworkConfig _config = new();

    public NeuralNetworkBuilder<TArch> WithArchitecture(int inputSize, int[] hiddenLayers, int outputSize)
    {
        _config.Architecture = [inputSize, .. hiddenLayers, outputSize];

        return this;
    }

    public NeuralNetworkBuilder<TArch> WithEpochs(int epochs)
    {
        _config.Epochs = epochs;

        return this;
    }

    public NeuralNetworkBuilder<TArch> WithBatchSize(int batchSize)
    {
        _config.BatchSize = batchSize;

        return this;
    }

    public NeuralNetworkBuilder<TArch> WithLearningRate(float rate)
    {
        _config.LearningRate = rate;

        return this;
    }

    public NeuralNetworkBuilder<TArch> WithWeightDecay(float decay)
    {
        _config.WeightDecay = decay;

        return this;
    }

    public NeuralNetworkBuilder<TArch> WithModel(IModel model)
    {
        _config.Model = model;

        return this;
    }

    public NeuralNetworkBuilder<TArch> WithShuffle(bool shuffle)
    {
        _config.WithShuffle = shuffle;

        return this;
    }

    public NeuralNetwork<TArch> Build()
    {
        if (_config.Architecture == null || _config.Architecture.Length == 0)
        {
            throw new InvalidOperationException("Architecture must be specified");
        }

        if (_config.Model == null)
        {
            throw new InvalidOperationException("Model must be specified");
        }

        return new NeuralNetwork<TArch>(_config);
    }
}
