using System;
using System.IO;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public class CnnNetwork
{
    private readonly CnnNeuralFramework _framework;

    public CnnNetwork(CnnNeuralFramework framework)
    {
        _framework = framework;
    }

    public float TrainBatch(CnnMatrix input, NeuralMatrix target, float learningRate)
    {
        return _framework.Train(input, target, learningRate);
    }

    public NeuralMatrix Forward(CnnMatrix input)
    {
        return _framework.Forward(input);
    }

    #region Save and Load Methods

    public CnnMatrix[] GetConvLayerOutput(CnnMatrix input)
    {
        return _framework.GetConvLayerOutput(input);
    }

    /// <summary>
    /// Saves weights to a binary stream using an enum key.
    /// </summary>
    public CnnNetwork SaveData<TEnum>(TEnum key, Stream stream) where TEnum : struct, Enum
    {
        _framework.SaveData(key, stream);
        return this;
    }

    /// <summary>
    /// Saves weights to a directory path using an enum key.
    /// </summary>
    public CnnNetwork SaveData<TEnum>(TEnum key, string directoryPath) where TEnum : struct, Enum
    {
        _framework.SaveData(key, directoryPath);
        return this;
    }

    /// <summary>
    /// Loads weights from a binary stream using an enum key.
    /// </summary>
    public CnnNetwork LoadData<TEnum>(TEnum key, Stream stream) where TEnum : struct, Enum
    {
        if (!_framework.LoadData(key, stream)) throw new InvalidOperationException();

        return this;
    }

    /// <summary>
    /// Loads weights from a directory path using an enum key.
    /// </summary>
    public bool LoadData<TEnum>(TEnum key, string directoryPath) where TEnum : struct, Enum
    {
        return _framework.LoadData(key, directoryPath);
    }

    #endregion

    public void Dispose()
    {
        _framework.Dispose();
    }
}
