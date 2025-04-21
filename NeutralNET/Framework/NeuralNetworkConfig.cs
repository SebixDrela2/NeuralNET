using NeutralNET.Models;

namespace NeutralNET.Framework;

public class NeuralNetworkConfig
{
    public int[] Architecture { get; set; }
    public int Epochs { get; set; } = 2000;
    public int BatchSize { get; set; } = 100;
    public float LearningRate { get; set; } = 1e-2f;
    public float WeightDecay { get; set; } = 1e-4f;
    public IModel Model { get; set; }
}
