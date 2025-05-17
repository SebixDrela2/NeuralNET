using NeutralNET.Attributes;
using NeutralNET.Matrices;
using NeutralNET.Models;

namespace NeutralNET.Framework;

[ArchitectureConfig(InputSize = DigitModel.PixelCount, HiddenLayers = [64, 64, 64], OutputSize = 1)]
public partial class Architecture : IArchitecture<Architecture>
{
    public NeuralMatrix[] MatrixNeurons { get; }
    public NeuralMatrix[] MatrixWeights { get; }
    public NeuralMatrix[] MatrixBiases { get; }

    public NeuralMatrix[] MatrixMWeights { get; }
    public NeuralMatrix[] MatrixVWeights { get; }
    public NeuralMatrix[] MatrixMBiases { get; } 
    public NeuralMatrix[] MatrixVBiases { get; }
    
    public int Count { get; }

    public Architecture(params ReadOnlySpan<int> architecture)
    {
        Count = architecture.Length - 1;

        MatrixNeurons = new NeuralMatrix[architecture.Length];
        MatrixWeights = new NeuralMatrix[Count];
        MatrixBiases = new NeuralMatrix[Count];

        MatrixMWeights = new NeuralMatrix[Count];
        MatrixVWeights = new NeuralMatrix[Count];
        MatrixMBiases = new NeuralMatrix[Count];
        MatrixVBiases = new NeuralMatrix[Count];

        MatrixNeurons[0] = new NeuralMatrix(1, architecture[0]);

        for (var i = 1; i < architecture.Length; i++)
        {
            var layerIndex = i - 1;

            MatrixWeights[layerIndex] = new NeuralMatrix(
                rows: architecture[i],
                columns: MatrixNeurons[i - 1].UsedColumns
            );
            MatrixBiases[layerIndex] = new NeuralMatrix(1, architecture[i]);
            MatrixNeurons[i] = new NeuralMatrix(1, architecture[i]);
            
            MatrixMWeights[layerIndex] = new NeuralMatrix(
                MatrixWeights[layerIndex].Rows,
                MatrixWeights[layerIndex].UsedColumns
            );
            MatrixVWeights[layerIndex] = new NeuralMatrix(
                MatrixWeights[layerIndex].Rows,
                MatrixWeights[layerIndex].UsedColumns
            );
            MatrixMBiases[layerIndex] = new NeuralMatrix(
                MatrixBiases[layerIndex].Rows,
                MatrixBiases[layerIndex].UsedColumns
            );
            MatrixVBiases[layerIndex] = new NeuralMatrix(
                MatrixBiases[layerIndex].Rows,
                MatrixBiases[layerIndex].UsedColumns
            );
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
