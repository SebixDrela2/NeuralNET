using NeutralNET.Framework.Neural.CNN;

namespace NeutralNET.Framework.Convolutional;

public static class CnnOptimizerFactory
{
    public static ICnnOptimizer Create(
        CnnOptimizerConfig config,
        OptimizerHyperLayerParameterSet convHyperParameters,
        OptimizerHyperLayerParameterSet denseHyperParameters)
        => config.OptimizerType switch
        {
            CnnOptimizerType.SGD => new CnnSGDOptimizer(config, convHyperParameters, denseHyperParameters),
            CnnOptimizerType.Adam => new CnnAdamOptimizer(config, convHyperParameters, denseHyperParameters),
            CnnOptimizerType.AdamW => new CnnAdamWOptimizer(config, convHyperParameters, denseHyperParameters),
            _ => throw new NotSupportedException($"Optimizer {config.OptimizerType} not supported.")
        };
}
