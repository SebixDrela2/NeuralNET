using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using NeutralNET.Stuff;

namespace NeutralNET.Test.Data;

public class DigiDigiDataLoader : DataLoaderBase
{
    private static readonly string[] _fontNames =
        ["Arial", "Times New Roman", "Georgia", "Verdana", "Tahoma"];

    public override int ImageScale => 16;
    public override string DatasetName => "DigiDigi";

    protected override (List<CnnMatrix> trainImages, List<NeuralMatrix> trainLabels,
                        List<CnnMatrix> testImages, List<NeuralMatrix> testLabels)
        LoadBatches(int batchSize, int maxTrainSamples, int maxTestSamples)
    {
        var (trainImages, trainLabels) = LoadFlattenedDataSet(DataSetType.Train, batchSize, maxTrainSamples);
        var (testImages, testLabels) = LoadFlattenedDataSet(DataSetType.Test, batchSize, maxTestSamples);

        return (trainImages, trainLabels, testImages, testLabels);
    }

    protected override void AddToBatches(float[][] images, int[] labels, int batchSize,
                                         List<CnnMatrix> outImages, List<NeuralMatrix> outLabels)
    {
        int scale = ImageScale;
        int numSamples = images.Length;

        for (int start = 0; start < numSamples; start += batchSize)
        {
            int end = Math.Min(start + batchSize, numSamples);
            int currentBatchSize = end - start;

            var imgMat = CnnMatrix.GetOrCreate(currentBatchSize, Channels, scale, scale, readOnly: true);
            var lblMat = NeuralMatrix.GetOrCreate(currentBatchSize, NumClasses);

            for (int i = 0; i < currentBatchSize; i++)
            {
                int idx = start + i;
                float[] pixels = images[idx];

                for (int c = 0; c < Channels; c++)
                {
                    int offset = c * scale * scale;
                    for (int y = 0; y < scale; y++)
                    {
                        for (int x = 0; x < scale; x++)
                        {
                            imgMat[i, c, y, x] = pixels[offset + y * scale + x];
                        }
                    }
                }

                int label = labels[idx];
                lblMat.Set(i, label, 1.0f);
            }

            outImages.Add(imgMat);
            outLabels.Add(lblMat);
        }
    }

    private (List<CnnMatrix> images, List<NeuralMatrix> labels) LoadFlattenedDataSet(
        DataSetType dataSetType, int batchSize, int maxSamples)
    {
        var batchImages = new List<CnnMatrix>();
        var batchLabels = new List<NeuralMatrix>();
        int samplesLoaded = 0;
        var fontNames = _fontNames.ToArray();
        int batchCount = dataSetType == DataSetType.Train ? 5 : 1;

        for (int i = 1; i <= batchCount; i++)
        {
            if (samplesLoaded >= maxSamples)
                break;

            Random.Shared.Shuffle(fontNames);

            var data = fontNames
                .SelectMany(font => GraphicsUtils.GetDigitsDataSetRGB(font))
                .Skip(samplesLoaded)
                .Take(maxSamples - samplesLoaded);

            var (labelArray, imageArray) = (
                data.Select(x => x.Label).ToArray(),
                data.Select(x => x.Flat.ToArray()).ToArray()
            );

            AddToBatches(imageArray, labelArray, batchSize, batchImages, batchLabels);
            samplesLoaded += imageArray.Length;
        }

        Console.WriteLine($"{DatasetName}: Loaded {samplesLoaded} {dataSetType} samples in {batchImages.Count} batches");

        var flatImages = FlattenBatchesToImages(batchImages);
        var flatLabels = FlattenBatchesToLabels(batchLabels);

        foreach (var img in batchImages) img?.Dispose();
        foreach (var lbl in batchLabels) lbl?.Dispose();

        return (flatImages, flatLabels);
    }

    private List<CnnMatrix> FlattenBatchesToImages(List<CnnMatrix> batches)
    {
        int scale = ImageScale;
        var images = new List<CnnMatrix>();

        foreach (var batch in batches)
        {
            for (int i = 0; i < batch.Batch; i++)
            {
                var single = CnnMatrix.GetOrCreate(1, Channels, scale, scale);
                for (int c = 0; c < Channels; c++)
                    for (int y = 0; y < scale; y++)
                        for (int x = 0; x < scale; x++)
                            single[0, c, y, x] = batch[i, c, y, x];
                images.Add(single);
            }
        }
        return images;
    }

    private List<NeuralMatrix> FlattenBatchesToLabels(List<NeuralMatrix> batches)
    {
        var labels = new List<NeuralMatrix>();

        foreach (var batch in batches)
        {
            for (int i = 0; i < batch.Rows; i++)
            {
                var single = NeuralMatrix.GetOrCreate(1, NumClasses);
                for (int j = 0; j < NumClasses; j++)
                    single.At(0, j) = batch.At(i, j);
                labels.Add(single);
            }
        }
        return labels;
    }

    private enum DataSetType
    {
        Train,
        Test
    }
}
