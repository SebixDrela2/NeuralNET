namespace NeutralNET.Framework;

public class NeuralNetwork
{
    private NeuralNetworkConfig _config;

    public NeuralNetwork(NeuralNetworkConfig config)
    {
        _config = config;
    }

    public IModelRunner Run()
    {
        var neuralFramework = new NeuralFramework(_config);
        var gradientFramework = new NeuralFramework(_config);

        return neuralFramework.Run(gradientFramework, _config.Model);
    }
}
