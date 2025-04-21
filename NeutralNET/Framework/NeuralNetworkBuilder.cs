using NeutralNET.Models;

namespace NeutralNET.Framework;

public class NeuralNetworkBuilder
{
    private readonly NeuralNetworkConfig _config = new();

    public NeuralNetworkBuilder WithArchitecture(params int[] architecture)
    {
        _config.Architecture = architecture;

        return this;
    }

    public NeuralNetworkBuilder WithEpochs(int epochs)
    {
        _config.Epochs = epochs;

        return this;
    }

    public NeuralNetworkBuilder WithBatchSize(int batchSize)
    {
        _config.BatchSize = batchSize;

        return this;
    }

    public NeuralNetworkBuilder WithLearningRate(float rate)
    {
        _config.LearningRate = rate;

        return this;
    }

    public NeuralNetworkBuilder WithWeightDecay(float decay)
    {
        _config.WeightDecay = decay;

        return this;
    }

    public NeuralNetworkBuilder WithModel(IModel model)
    {
        _config.Model = model;

        return this;
    }

    public NeuralNetworkBuilder WithShuffle(bool shuffle)
    {
        _config.WithShuffle = shuffle;

        return this;
    }

    public NeuralNetwork Build()
    {
        if (_config.Architecture == null || _config.Architecture.Length == 0)
        {
            throw new InvalidOperationException("Architecture must be specified");
        }

        if (_config.Model == null)
        {
            throw new InvalidOperationException("Model must be specified");
        }

        return new NeuralNetwork(_config);
    }
}
