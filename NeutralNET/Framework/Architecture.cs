using NeutralNET.Matrices;
using System.Runtime.InteropServices;

namespace NeutralNET.Framework;

internal class Architecture
{
    public NeuralMatrix[] MatrixNeurons = null!;
    public NeuralMatrix[] MatrixWeights = null!;
    public NeuralMatrix[] MatrixBiases = null!;

    public readonly int Count;

    public Architecture(int[] architecture)
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
