using NeutralNET.Framework.Neural;

namespace NeutralNET.Framework.Optimizers;

internal class OptimizerFactory<TArch>(
    NeuralNetworkConfig config, 
    TArch architecture, 
    TArch gradientArchitecture) where TArch : IArchitecture<TArch>
{
    public IOptimizer GetOptimizer() => config.OptimizerType switch
    {
        OptimizerType.SGD => new SGDOptimizer<TArch>(config, architecture, gradientArchitecture),
        OptimizerType.Adam => new AdamOptimizer<TArch>(config, architecture, gradientArchitecture),
        _ => throw new NotImplementedException("Unrecognized optimizer")
    };
}
