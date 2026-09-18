using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;

namespace NeutralNET.Framework.Neural.CNN;

public class CnnBuilder 
{
    private NeuralNetworkConfig _denseConfig;
    private CnnArchitectureConfig _cnnConfig;

    private int _inputBatchSize;
    private int _inputChannels;
    private int _inputHeight;
    private int _inputWidth;

    public CnnBuilder WithDenseConfig(NeuralNetworkConfig config)
    {
        _denseConfig = config;
        return this;
    }

    public CnnBuilder WithCnnConfig(CnnArchitectureConfig config)
    {
        _cnnConfig = config;
        return this;
    }

    public CnnBuilder WithInputSize(int batchSize, int channels, int height, int width)
    {
        _inputBatchSize = batchSize;
        _inputChannels = channels;
        _inputHeight = height;
        _inputWidth = width;
        return this;
    }

    public CnnNetwork Build()
    {
        if (_denseConfig == null)
            throw new InvalidOperationException("DenseConfig must be set before building.");
        if (_cnnConfig == null)
            throw new InvalidOperationException("CnnConfig must be set before building.");

        var framework = new CnnNeuralFramework(
            _denseConfig,
            _cnnConfig,
            _inputBatchSize,
            _inputChannels,
            _inputHeight,
            _inputWidth);

        return new CnnNetwork(framework);
    }
}
