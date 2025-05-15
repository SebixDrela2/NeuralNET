using NeutralNET.Matrices;

namespace NeutralNET.Framework;

public class NeuralNetwork<TArch> where TArch : IArchitecture<TArch>
{
    private NeuralNetworkConfig _config;

    public NeuralNetwork(NeuralNetworkConfig config)
    {
        _config = config;
    }

    public NeuralForward Run()
    {
        var neuralFramework = new NeuralFramework<TArch>(_config);

        return neuralFramework.Run(_config.Model);
    }
}
