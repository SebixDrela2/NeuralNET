using NeutralNET.Framework.Neural.CNN;

namespace NeutralNET.Framework.Convolutional;

public static class CnnOptimizerFactory
{
    public static ICnnOptimizer Create(CnnOptimizerConfig config)
    {
        return config.OptimizerType switch
        {
            CnnOptimizerType.SGD => new CnnSGDOptimizer(config),
            CnnOptimizerType.Adam => new CnnAdamOptimizer(config),
            _ => throw new NotSupportedException($"Optimizer {config.OptimizerType} not supported.")
        };
    }
}
