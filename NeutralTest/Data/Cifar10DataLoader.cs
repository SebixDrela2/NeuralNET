using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Test.Data;

public class Cifar10DataLoader : DataLoaderBase
{
    private const string DataDir = @"D:\Cifar\datasets\cifar-10-batches-bin";

    public override int ImageScale => 32;
    public override string DatasetName => "CIFAR-10";

    protected override (List<CnnMatrix> trainImages, List<NeuralMatrix> trainLabels,
                        List<CnnMatrix> testImages, List<NeuralMatrix> testLabels)
        LoadBatches(int batchSize, int maxTrainSamples, int maxTestSamples)
    {
        var trainImages = new List<CnnMatrix>();
        var trainLabels = new List<NeuralMatrix>();
        var testImages = new List<CnnMatrix>();
        var testLabels = new List<NeuralMatrix>();

        int trainSamplesLoaded = 0;
        int testSamplesLoaded = 0;

        for (int i = 1; i <= 5; i++)
        {
            if (trainSamplesLoaded >= maxTrainSamples)
                break;

            string path = Path.Combine(DataDir, $"data_batch_{i}.bin");
            if (!File.Exists(path))
                throw new FileNotFoundException($"Training batch not found: {path}");

            var (images, labels) = ReadCifarBatch(path);

            int samplesToTake = Math.Min(images.Length, maxTrainSamples - trainSamplesLoaded);
            if (samplesToTake < images.Length)
            {
                images = images.Take(samplesToTake).ToArray();
                labels = labels.Take(samplesToTake).ToArray();
            }

            AddToBatches(images, labels, batchSize, trainImages, trainLabels);
            trainSamplesLoaded += images.Length;
        }

        string testPath = Path.Combine(DataDir, "test_batch.bin");
        if (File.Exists(testPath))
        {
            var (testImgs, testLbls) = ReadCifarBatch(testPath);

            int samplesToTake = Math.Min(testImgs.Length, maxTestSamples);
            if (samplesToTake < testImgs.Length)
            {
                testImgs = testImgs.Take(samplesToTake).ToArray();
                testLbls = testLbls.Take(samplesToTake).ToArray();
            }

            AddToBatches(testImgs, testLbls, batchSize, testImages, testLabels);
            testSamplesLoaded += testImgs.Length;
        }

        Console.WriteLine($"{DatasetName}: Loaded {trainSamplesLoaded} training samples in {trainImages.Count} batches");
        Console.WriteLine($"{DatasetName}: Loaded {testSamplesLoaded} test samples in {testImages.Count} batches");

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

    private (float[][] images, int[] labels) ReadCifarBatch(string path)
    {
        int imageSize = ImageScale * ImageScale * Channels;
        byte[] data = File.ReadAllBytes(path);
        int numSamples = data.Length / (imageSize + 1);

        var images = new float[numSamples][];
        var labels = new int[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            int offset = i * (imageSize + 1);
            labels[i] = data[offset];

            float[] img = new float[imageSize];
            for (int j = 0; j < imageSize; j++)
                img[j] = data[offset + 1 + j] / 255.0f;
            images[i] = img;
        }

        return (images, labels);
    }
}
