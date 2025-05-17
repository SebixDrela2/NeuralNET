using NeutralNET.Models;

namespace NeutralNET.Framework.Neural;

public class NeuralNetworkConfig
{
    public int[] Architecture { get; set; } = null!;
    public int Epochs { get; set; } = 2000;
    public int BatchSize { get; set; } = 100;
    public float LearningRate { get; set; } = 1e-2f;
    public float WeightDecay { get; set; } = 1e-4f;
    public bool WithShuffle { get; set; }
    public IModel Model { get; set; } = null!;

    public float Beta1 { get; set; } = 0.9f;
    public float Beta2 { get; set; } = 0.999f;
    public float Epsilon { get; set; } = 1e-8f;
}
