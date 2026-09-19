using System.Data;
using System.Runtime.Versioning;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using NeutralTest;

namespace NeutralNET.Test.Data;

public abstract class DataLoaderBase
{
    // Changed from "protected const int" to a virtual property so derived classes can override it
    public virtual int NumClasses => 10;

    protected const int Channels = 3;

    /// <summary>
    /// Gets the image scale (width/height) for this dataset
    /// </summary>
    public abstract int ImageScale { get; }

    /// <summary>
    /// Gets the name of the dataset
    /// </summary>
    public abstract string DatasetName { get; }

    /// <summary>
    /// Loads the complete dataset
    /// </summary>
    ///
    public NeuralDataset LoadCompleteDataset(int batchSize = 64, int maxTrainSamples = int.MaxValue, int maxTestSamples = int.MaxValue)
    {
        var (trainImages, trainLabels, testImages, testLabels) = LoadBatches(batchSize, maxTrainSamples, maxTestSamples);

        return new NeuralDataset
        {
            TrainImages = trainImages,
            TrainLabels = trainLabels,
            TrainActualLabels = ExtractActualLabels(trainLabels),
            TestImages = testImages,
            TestLabels = testLabels,
            TestActualLabels = ExtractActualLabels(testLabels)
        };
    }

    ///<summary>
    /// Reloads complete dataset.
    ///</summary>
    public void ReloadCompleteDataset(NeuralDataset dataset, CnnTrainingConfig config)
    {
        var (trainImages, trainLabels, testImages, testLabels) = LoadBatches(config.BatchSize, config.MaxTrainSamples, config.MaxTestSamples);

        for (int i = 0; i < trainImages.Count; i++) { dataset.TrainImages[i].CopyFrom(trainImages[i]); trainImages[i].Dispose(); }
        for (int i = 0; i < trainLabels.Count; i++) { dataset.TrainLabels[i].CopyFrom(trainLabels[i]); trainLabels[i].Dispose(); }
        for (int i = 0; i < testImages.Count; i++) { dataset.TestImages[i].CopyFrom(testImages[i]); testImages[i].Dispose(); }
        for (int i = 0; i < testLabels.Count; i++) { dataset.TestLabels[i].CopyFrom(testLabels[i]); testLabels[i].Dispose(); }
    }

    /// <summary>
    /// Loads the raw batches (to be implemented by derived classes)
    /// </summary>
    protected abstract (List<CnnMatrix> trainImages, List<NeuralMatrix> trainLabels,
                        List<CnnMatrix> testImages, List<NeuralMatrix> testLabels)
        LoadBatches(int batchSize, int maxTrainSamples, int maxTestSamples);

    /// <summary>
    /// Adds image data and labels to batch lists (to be implemented by derived classes)
    /// </summary>
    protected abstract void AddToBatches(float[][] images, int[] labels, int batchSize,
                                         List<CnnMatrix> outImages, List<NeuralMatrix> outLabels);

    // ============================================================================
    // SHARED HELPER METHODS
    // ============================================================================

    protected static int[] ExtractActualLabels(List<NeuralMatrix> labelBatches)
    {
        if (labelBatches is not [_, ..]) return [];

        int totalSamples = labelBatches.Sum(l => l.Rows);
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
        var combined = CnnMatrix.GetOrCreate(totalBatch, Channels, ImageScale, ImageScale);

        int offset = 0;
        foreach (var mat in matrices)
        {
            if (mat == null) continue;
            for (int b = 0; b < mat.Batch; b++)
            {
                for (int c = 0; c < Channels; c++)
                {
                    for (int y = 0; y < ImageScale; y++)
                    {
                        for (int x = 0; x < ImageScale; x++)
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
}
