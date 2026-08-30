namespace NeutralNET.Framework.Convolutional;

public class CnnOptimizerConfig
{
    public CnnOptimizerType OptimizerType { get; set; } = CnnOptimizerType.SGD;
    public float LearningRate { get; set; } = 0.01f;
    public float WeightDecay { get; set; } = 0.0005f;
    public float Momentum { get; set; } = 0.9f;          // for SGD
    public float Beta1 { get; set; } = 0.9f;             // for Adam
    public float Beta2 { get; set; } = 0.999f;
    public float Epsilon { get; set; } = 1e-8f;
}
