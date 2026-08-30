using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public class CnnNetwork<TArch> : IDisposable where TArch : IArchitecture<TArch>
{
    private readonly CnnNeuralFramework<TArch> _framework;

    public CnnNetwork(CnnNeuralFramework<TArch> framework)
    {
        _framework = framework;
    }

    public CnnNetwork<TArch> Train(CnnMatrix input, NeuralMatrix target, float learningRate)
    {
        _framework.Train(input, target, learningRate);
        return this;
    }

    public CnnNetwork<TArch> TrainEpoch(List<CnnMatrix> inputs, List<NeuralMatrix> targets, float learningRate)
    {
        for (int i = 0; i < inputs.Count; i++)
        {
            _framework.Train(inputs[i], targets[i], learningRate);
        }
        return this;
    }

    public float TrainBatch(CnnMatrix input, NeuralMatrix target, float learningRate)
    {
        return _framework.Train(input, target, learningRate);
    }

    public NeuralMatrix Forward(CnnMatrix input)
    {
        return _framework.Forward(input);
    }

    public void Dispose()
    {
        _framework.Dispose();
    }
}
