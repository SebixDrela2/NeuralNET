using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public class CnnBuilder<TArch> where TArch : IArchitecture<TArch>
{
    private NeuralNetworkConfig _denseConfig;
    private CnnArchitectureConfig _cnnConfig;
    private int _inputHeight = 32;
    private int _inputWidth = 32;
    private int _inputChannels = 3;

    public CnnBuilder<TArch> WithDenseConfig(NeuralNetworkConfig config)
    {
        _denseConfig = config;
        return this;
    }

    public CnnBuilder<TArch> WithCnnConfig(CnnArchitectureConfig config)
    {
        _cnnConfig = config;
        return this;
    }

    public CnnBuilder<TArch> WithInputSize(int height, int width, int channels = 3)
    {
        _inputHeight = height;
        _inputWidth = width;
        _inputChannels = channels;
        return this;
    }

    public CnnNetwork<TArch> Build()
    {
        if (_denseConfig == null)
            throw new InvalidOperationException("DenseConfig must be set before building.");
        if (_cnnConfig == null)
            throw new InvalidOperationException("CnnConfig must be set before building.");

        var framework = new CnnNeuralFramework<TArch>(
            _denseConfig,
            _cnnConfig,
            _inputHeight,
            _inputWidth,
            _inputChannels
        );

        return new CnnNetwork<TArch>(framework);
    }
}
