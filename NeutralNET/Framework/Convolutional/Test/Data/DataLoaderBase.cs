using System.Data;
using System.Numerics;
using System.Runtime.Versioning;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using NeutralNET.Utils;
using NeutralTest;

namespace NeutralNET.Test.Data;

public interface IDataLoader
{
    DataSourceType DataSourceType { get; }
}

public interface IDataLoader<TSelf> : IDataLoader
    where TSelf : DataLoaderBase, IDataLoader<TSelf>
{
    static new abstract DataSourceType DataSourceType { get; }
    static abstract TSelf Create();

    DataSourceType IDataLoader.DataSourceType => TSelf.DataSourceType;
}

public abstract class DataLoaderBase : IDisposable
{
    public virtual int NumClasses => 10;

    public const int Channels = 3;
    public abstract int ImageWidth { get; }
    public abstract int ImageHeight { get; }

    /// <summary>
    /// Gets the name of the dataset
    /// </summary>
    public abstract string DatasetName { get; }

    public enum DataSetType
    {
        Train,
        Test,
    }

    /// <summary>
    /// Loads the complete dataset
    /// </summary>
    public NeuralDataSet LoadCompleteDataset(int batchSize, int maxTrainSamples, int maxTestSamples)
    {
        return LoadBatches(batchSize, (maxTrainSamples, maxTestSamples));
    }

    ///<summary>
    /// Reloads complete dataset.
    ///</summary>
    public void ReloadCompleteDataset(NeuralDataSet dataset, CnnTrainingConfig config)
    {
        UpdateBatches(config.BatchSize, (config.MaxTrainSamples, config.MaxTestSamples), dataset);
    }

    public abstract NeuralTrainSet CreateTrainSet(DataSetType dataSetType, int batchSize, int maxSamples);
    public abstract void UpdateTrainSet(DataSetType dataSetType, int batchSize, int maxSamples, NeuralTrainSet output);

    /// <summary>
    /// Loads the raw batches (to be implemented by derived classes)
    /// </summary>
    protected abstract NeuralDataSet LoadBatches(int batchSize, (int Train, int Test) samples);

    protected abstract void UpdateBatches(int batchSize, (int Train, int Test) samples, NeuralDataSet dataSet);

    /// <summary>
    /// Adds image data and labels to batch lists (to be implemented by derived classes)
    /// </summary>
    protected abstract void AddToBatches(float[][] images, int[] labels, int batchSize,
                                         List<CnnMatrix> outImages, List<NeuralMatrix> outLabels);

    // ============================================================================
    // SHARED HELPER METHODS
    // ============================================================================

    protected static int[] ExtractActualLabels(ReadOnlySpan<NeuralMatrix> labelBatches)
    {
        if (labelBatches is not [_, ..]) return [];

        int totalSamples = labelBatches.SumBy(l => l.Rows);
        int[] actualLabels = new int[totalSamples];

        int offset = 0;
        foreach (var lbl in labelBatches)
        {
            for (int i = 0; i < lbl.Rows; i++) actualLabels[offset + i] = ArgMax(lbl.GetRowSpan(i));
            offset += lbl.Rows;
        }

        return actualLabels;
    }

    protected static int ArgMax(Span<float> row)
    {
        var max = (Index: 0, Value: row[0]);
        for (int i = 1; i < row.Length; i++)
        {
            if (row[i] > max.Value) max = (i, row[i]);
        }
        return max.Index;
    }

    protected CnnMatrix CombineCnnMatrices(List<CnnMatrix> matrices)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(matrices.Count);
        if (matrices is [var single]) return single;

        int totalBatch = matrices.Sum(m => m.Batch);
        var combined = CnnMatrix.GetOrCreate(totalBatch, Channels, ImageHeight, ImageWidth);

        int offset = 0;
        foreach (var mat in matrices)
        {
            if (mat == null) continue;
            for (int b = 0; b < mat.Batch; b++)
            {
                for (int c = 0; c < Channels; c++)
                {
                    for (int y = 0; y < ImageHeight; y++)
                    {
                        for (int x = 0; x < ImageWidth; x++)
                        {
                            combined[offset + b, c, y, x] = mat[b, c, y, x];
                        }
                    }
                }
            }
            offset += mat.Batch;
        }

        return combined;
    }

    protected NeuralMatrix CombineNeuralMatrices(List<NeuralMatrix> matrices)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(matrices.Count);
        if (matrices is [var single]) return single;

        int totalRows = matrices.Sum(m => m.Rows);
        var combined = NeuralMatrix.GetOrCreate(totalRows, NumClasses);

        int offset = 0;
        foreach (var mat in matrices)
        {
            for (int i = 0; i < mat.Rows; i++)
            {
                for (int j = 0; j < NumClasses; j++)
                {
                    combined.At(offset + i, j) = mat.At(i, j);
                }
            }
            offset += mat.Rows;
        }

        return combined;
    }

    public virtual void Dispose() { }
}
