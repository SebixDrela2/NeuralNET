using NeutralNET.Activation;
using NeutralNET.Framework.Neural.CNN;

public class CnnArchitectureConfig
{
    public List<CnnLayerConfig> ConvLayers { get; set; } = [];
    public int[] DenseArchitecture { get; set; } = [];
    public ActivationType DenseHiddenActivation { get; set; } = ActivationType.ReLU;
    public ActivationType OutputActivation { get; set; } = ActivationType.Softmax; 
}
