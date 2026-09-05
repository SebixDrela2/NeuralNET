using NeutralNET.Activation;
using NeutralNET.Matrices;
using static NeutralNET.Activation.ActivationSelector;

namespace NeutralNET.GPU;

/// <summary>
/// CPU fallback backend (uses existing AVX implementations)
/// </summary>
public class CpuBackend : IGpuBackend
{
    private readonly ActivationSelector _activationSelector = new();

    public bool IsAvailable => true;
    public string DeviceName => "CPU (AVX optimized)";

    public NeuralMatrix Multiply(NeuralMatrix a, NeuralMatrix b)
    {
        var result = NeuralMatrix.GetOrCreate(a.Rows, b.UsedColumns);
        a.DotVectorized(b, result);
        return result;
    }

    public void Multiply(NeuralMatrix a, NeuralMatrix b, NeuralMatrix result)
    {
        a.DotVectorized(b, result);
    }

    public NeuralMatrix Add(NeuralMatrix a, NeuralMatrix b)
    {
        var result = NeuralMatrix.GetOrCreate(a.Rows, a.UsedColumns);
        result.CopyDataFrom(a);
        result.SumVectorized(b);
        return result;
    }

    public void Add(NeuralMatrix a, NeuralMatrix b, NeuralMatrix result)
    {
        result.CopyDataFrom(a);
        result.SumVectorized(b);
    }

    public NeuralMatrix Transpose(NeuralMatrix matrix)
    {
        var result = NeuralMatrix.GetOrCreate(matrix.UsedColumns, matrix.Rows);
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.UsedColumns; j++)
            {
                result.At(j, i) = matrix.At(i, j);
            }
        }
        return result;
    }

    public void ApplyActivation(NeuralMatrix matrix, ActivationType activation)
    {
        var act = _activationSelector.GetActivation(activation);
        act(matrix);
    }

    public void Softmax(NeuralMatrix matrix) => ActivationFunctions.ApplySoftmaxVectorized(matrix);

    public void CopyToDevice(NeuralMatrix matrix) { }
    public void CopyToHost(NeuralMatrix matrix) { }
    public void Synchronize() { }
    public void Dispose() { }
}
