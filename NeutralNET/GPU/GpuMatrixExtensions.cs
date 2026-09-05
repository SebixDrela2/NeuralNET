using global::NeutralNET.Matrices;

namespace NeutralNET.GPU;

/// <summary>
/// Extension methods for GPU-accelerated matrix operations
/// </summary>
public static class GpuMatrixExtensions
{
    private static readonly IGpuBackend _backend = GpuBackendFactory.Instance;

    /// <summary>
    /// Multiplies two matrices using GPU if available
    /// </summary>
    public static NeuralMatrix GpuMultiply(this NeuralMatrix a, NeuralMatrix b)
    {
        return _backend.Multiply(a, b);
    }

    /// <summary>
    /// Multiplies two matrices using GPU if available, storing result in pre-allocated matrix
    /// </summary>
    public static void GpuMultiply(this NeuralMatrix a, NeuralMatrix b, NeuralMatrix result)
    {
        _backend.Multiply(a, b, result);
    }

    /// <summary>
    /// Adds two matrices using GPU if available
    /// </summary>
    public static NeuralMatrix GpuAdd(this NeuralMatrix a, NeuralMatrix b)
    {
        return _backend.Add(a, b);
    }

    /// <summary>
    /// Adds two matrices using GPU if available, storing result in pre-allocated matrix
    /// </summary>
    public static void GpuAdd(this NeuralMatrix a, NeuralMatrix b, NeuralMatrix result)
    {
        _backend.Add(a, b, result);
    }

    /// <summary>
    /// Transposes a matrix using GPU if available
    /// </summary>
    public static NeuralMatrix GpuTranspose(this NeuralMatrix matrix)
    {
        return _backend.Transpose(matrix);
    }

    /// <summary>
    /// Checks if GPU is available
    /// </summary>
    public static bool IsGpuAvailable => _backend.IsAvailable;

    /// <summary>
    /// Gets the GPU device name
    /// </summary>
    public static string GpuDeviceName => _backend.DeviceName;
}
