

using NeutralNET.Models;

namespace NeutralNET.Framework;

public class NeuralNetwork
{
    private int[] _architecture;
    private IModel _model;

    public NeuralNetwork(int[] architecture, IModel model)
    {
        _architecture = architecture;
        _model = model;
    }

    public IModelRunner Run()
    {
        var neuralFramework = new NeuralFramework(_architecture);
        var gradientFramework = new NeuralFramework(_architecture);

        return neuralFramework.Run(gradientFramework, _model);
    }
}
