using NeutralNET.Activation;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Neural.CNN;

public class CnnArchitectureConfig
{
    public List<CnnLayerConfig> ConvLayers { get; set; } = [];
    public int[] DenseArchitecture { get; set; } = [];
    public ActivationType DenseHiddenActivation { get; set; } = ActivationType.ReLU;
    public ActivationType OutputActivation { get; set; } = ActivationType.Softmax;
    public CnnOptimizerConfig OptimizerConfig { get; set; } = new CnnOptimizerConfig();
}
