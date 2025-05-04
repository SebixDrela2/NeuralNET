namespace NeutralNET.Framework;

public class NeuralNetwork
{
    private NeuralNetworkConfig _config;

    public NeuralNetwork(NeuralNetworkConfig config)
    {
        _config = config;
    }

    public void Run()
    {
        var neuralFramework = new NeuralFramework(_config);
        neuralFramework.Run(_config.Model);
    }
}
