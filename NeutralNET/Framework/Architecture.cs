using NeutralNET.Matrices;
using System.Runtime.InteropServices;

namespace NeutralNET.Framework;

internal class Architecture
{
    public Matrix[] MatrixNeurons = null!;
    public Matrix[] MatrixWeights = null!;
    public Matrix[] MatrixBiases = null!;

    public readonly int Count;

    public Architecture(int[] architecture)
    {
        Count = architecture.Length - 1;

        MatrixNeurons = new Matrix[architecture.Length];
        MatrixWeights = new Matrix[Count];
        MatrixBiases = new Matrix[Count];

        MatrixNeurons[0] = new Matrix(1, architecture[0]);

        for (var i = 1; i < architecture.Length; i++)
        {
            MatrixWeights[i - 1] = new Matrix(architecture[i], MatrixNeurons[i - 1].UsedColumns);
            MatrixBiases[i - 1] = new Matrix(1, architecture[i]);
            MatrixNeurons[i] = new Matrix(1, architecture[i]);
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
