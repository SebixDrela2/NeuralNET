using NeutralNET.Activation;
using NeutralNET.Matrices;

namespace NeutralNET.GPU;

/// <summary>
/// Interface for GPU-accelerated matrix operations
/// </summary>
public interface IGpuBackend : IDisposable
{
    bool IsAvailable { get; }
    string DeviceName { get; }

    /// <summary>
    /// Matrix multiplication: C = A * B
    /// </summary>
    NeuralMatrix Multiply(NeuralMatrix a, NeuralMatrix b);

    /// <summary>
    /// Matrix multiplication with result pre-allocated
    /// </summary>
    void Multiply(NeuralMatrix a, NeuralMatrix b, NeuralMatrix result);

    /// <summary>
    /// Element-wise addition: C = A + B
    /// </summary>
    NeuralMatrix Add(NeuralMatrix a, NeuralMatrix b);

    /// <summary>
    /// Element-wise addition with result pre-allocated
    /// </summary>
    void Add(NeuralMatrix a, NeuralMatrix b, NeuralMatrix result);

    /// <summary>
    /// Matrix transpose
    /// </summary>
    NeuralMatrix Transpose(NeuralMatrix matrix);

    /// <summary>
    /// Apply activation function to matrix
    /// </summary>
    void ApplyActivation(NeuralMatrix matrix, ActivationType activation);

    /// <summary>
    /// Apply softmax to matrix rows
    /// </summary>
    void Softmax(NeuralMatrix matrix);

    /// <summary>
    /// Copy data from CPU to GPU
    /// </summary>
    void CopyToDevice(NeuralMatrix matrix);

    /// <summary>
    /// Copy data from GPU to CPU
    /// </summary>
    void CopyToHost(NeuralMatrix matrix);

    /// <summary>
    /// Synchronize GPU operations
    /// </summary>
    void Synchronize();
}
