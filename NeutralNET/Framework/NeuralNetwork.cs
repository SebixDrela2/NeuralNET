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
        var gradientFramework = new NeuralFramework(_config);

        neuralFramework.Run(gradientFramework, _config.Model);
    }
}
