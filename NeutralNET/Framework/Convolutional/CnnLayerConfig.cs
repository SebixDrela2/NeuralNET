using NeutralNET.Activation;

namespace NeutralNET.Framework.Neural.CNN;

public class CnnLayerConfig
{
    public int KernelHeight { get; set; }
    public int KernelWidth { get; set; }
    public int Filters { get; set; }
    public int Stride { get; set; }
    public int Padding { get; set; }
    public ActivationType Activation { get; set; } = ActivationType.ReLU;
    public bool UseMaxPool { get; set; } = false;
    public int PoolSize { get; set; } = 2;
}
