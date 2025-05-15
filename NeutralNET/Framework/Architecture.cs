using NeutralNET.Matrices;

namespace NeutralNET.Framework;

public class Architecture : IArchitecture<Architecture>
{
    public NeuralMatrix[] MatrixNeurons { get; }
    public NeuralMatrix[] MatrixWeights { get; }
    public NeuralMatrix[] MatrixBiases { get; }

    public int Count { get; }

    public Architecture(params ReadOnlySpan<int> architecture)
    {
        Count = architecture.Length - 1;

        MatrixNeurons = new NeuralMatrix[architecture.Length];
        MatrixWeights = new NeuralMatrix[Count];
        MatrixBiases = new NeuralMatrix[Count];

        MatrixNeurons[0] = new NeuralMatrix(1, architecture[0]);

        for (var i = 1; i < architecture.Length; i++)
        {
            MatrixWeights[i - 1] = new NeuralMatrix(architecture[i], MatrixNeurons[i - 1].UsedColumns);
            MatrixBiases[i - 1] = new NeuralMatrix(1, architecture[i]);
            MatrixNeurons[i] = new NeuralMatrix(1, architecture[i]);
        }
    }

    public static Architecture Create(params ReadOnlySpan<int> architecture) => new(architecture);

    public void ZeroOut()
    {
        for (var i = 0; i < Count; i++)
        {
            MatrixNeurons[i].Clear();
            MatrixWeights[i].Clear();
            MatrixBiases[i].Clear();
        }
    }
}
