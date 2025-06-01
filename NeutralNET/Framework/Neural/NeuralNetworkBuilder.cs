using NeutralNET.Activation;
using NeutralNET.Framework.Optimizers;
using NeutralNET.Models;

namespace NeutralNET.Framework.Neural;

public class NeuralNetworkBuilder<TArch> where TArch : IArchitecture<TArch>
{
    private readonly NeuralNetworkConfig _config = new();

    public NeuralNetworkBuilder(IModel model)
    {
        _config.Model = model;
    }

    public NeuralNetworkBuilder<TArch> WithArchitecture(int[] hiddenLayers)
    {
        var inputUsedColumns = _config.Model.TrainingInput.UsedColumns;
        var outputUsedColumns = _config.Model.TrainingOutput.UsedColumns;

        _config.Architecture = [inputUsedColumns, .. hiddenLayers, outputUsedColumns];

        return this;
    }

    public NeuralNetworkBuilder<TArch> WithOptimizer(OptimizerType optimizerType)
    {
        _config.OptimizerType = optimizerType;

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

    public NeuralNetworkBuilder<TArch> WithHiddenLayerActivation(ActivationType activation)
    {
        _config.HiddenActivation = activation;

        return this;
    }

    public NeuralNetworkBuilder<TArch> WithOutputLayerActivation(ActivationType activation)
    {
        _config.OutputActivation = activation;
        return this;
    }

    public NeuralNetworkBuilder<TArch> WithWeightDecay(float decay)
    {
        _config.WeightDecay = decay;

        return this;
    }

    public NeuralNetworkBuilder<TArch> WithShuffle(bool shuffle)
    {
        _config.WithShuffle = shuffle;

        return this;
    }

    public NeuralNetworkBuilder<TArch> WithBeta1(float beta1)
    {
        _config.Beta1 = beta1;
        return this;
    }

    public NeuralNetworkBuilder<TArch> WithBeta2(float beta2)
    {
        _config.Beta2 = beta2;
        return this;
    }

    public NeuralNetworkBuilder<TArch> WithEpsilon(float epsilon)
    {
        _config.Epsilon = epsilon;
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
