using System.Data;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Test.Data;

public abstract class DataLoaderBase
{
    protected const int NumClasses = 10;
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
    public virtual NeuralDataset LoadCompleteDataset(int batchSize = 64, int maxTrainSamples = int.MaxValue, int maxTestSamples = int.MaxValue)
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

    /// <summary>
    /// Loads only the training split
    /// </summary>
    public virtual NeuralDataset LoadTrainingSplit(int batchSize = 64, int maxSamples = int.MaxValue)
    {
        var (trainImages, trainLabels, _, _) = LoadBatches(batchSize, maxSamples, 0);

        return new NeuralDataset
        {
            TrainImages = trainImages,
            TrainLabels = trainLabels,
            TrainActualLabels = ExtractActualLabels(trainLabels)
        };
    }

    /// <summary>
    /// Loads only the test split
    /// </summary>
    public virtual NeuralDataset LoadTestSplit(int batchSize = 64, int maxSamples = int.MaxValue)
    {
        var (_, _, testImages, testLabels) = LoadBatches(batchSize, 0, maxSamples);

        return new NeuralDataset
        {
            TrainImages = testImages,
            TrainLabels = testLabels,
            TrainActualLabels = ExtractActualLabels(testLabels)
        };
    }

    /// <summary>
    /// Convenience method that loads training data into a single batch
    /// </summary>
    public virtual (CnnMatrix images, NeuralMatrix labels, int[] actualLabels) LoadTrainingAsSingleBatch(int maxSamples = int.MaxValue)
    {
        var split = LoadTrainingSplit(maxSamples, maxSamples);
        var combinedImages = CombineCnnMatrices(split.TrainImages);
        var combinedLabels = CombineNeuralMatrices(split.TrainLabels);
        var actualLabels = ExtractActualLabels([combinedLabels]);

        split.Dispose();
        return (combinedImages, combinedLabels, actualLabels);
    }

    /// <summary>
    /// Returns an enumerable that yields training samples one at a time
    /// </summary>
    public virtual IEnumerable<(CnnMatrix image, NeuralMatrix label, int actualLabel)> EnumerateTrainingSamples(int maxSamples = int.MaxValue)
    {
        var (images, labels, actualLabels) = LoadTrainingAsSingleBatch(maxSamples);

        for (int i = 0; i < images.Batch; i++)
        {
            var singleImage = CnnMatrix.GetOrCreate(1, Channels, ImageScale, ImageScale);
            var singleLabel = NeuralMatrix.GetOrCreate(1, NumClasses);

            for (int c = 0; c < Channels; c++)
                for (int y = 0; y < ImageScale; y++)
                    for (int x = 0; x < ImageScale; x++)
                        singleImage[0, c, y, x] = images[i, c, y, x];

            for (int j = 0; j < NumClasses; j++)
                singleLabel.At(0, j) = labels.At(i, j);

            yield return (singleImage, singleLabel, actualLabels[i]);
        }

        images.Dispose();
        labels.Dispose();
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

    protected int[] ExtractActualLabels(List<NeuralMatrix> labelBatches)
    {
        if (labelBatches == null || labelBatches.Count == 0)
            return Array.Empty<int>();

        int totalSamples = labelBatches.Sum(l => l?.Rows ?? 0);
        int[] actualLabels = new int[totalSamples];
        int offset = 0;

        foreach (var lbl in labelBatches)
        {
            if (lbl == null) continue;
            for (int i = 0; i < lbl.Rows; i++)
                actualLabels[offset + i] = ArgMax(lbl.GetRowSpan(i));
            offset += lbl.Rows;
        }

        return actualLabels;
    }

    protected int ArgMax(Span<float> row)
    {
        int maxIdx = 0;
        float maxVal = row[0];
        for (int i = 1; i < row.Length; i++)
        {
            if (row[i] > maxVal)
            {
                maxVal = row[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    protected CnnMatrix CombineCnnMatrices(List<CnnMatrix> matrices)
    {
        if (matrices == null || matrices.Count == 0)
            return null;

        if (matrices.Count == 1)
            return matrices[0];

        int totalBatch = matrices.Sum(m => m?.Batch ?? 0);
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
        if (matrices == null || matrices.Count == 0)
            return null;

        if (matrices.Count == 1)
            return matrices[0];

        int totalRows = matrices.Sum(m => m?.Rows ?? 0);
        var combined = NeuralMatrix.GetOrCreate(totalRows, NumClasses);

        int offset = 0;
        foreach (var mat in matrices)
        {
            if (mat == null) continue;
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
