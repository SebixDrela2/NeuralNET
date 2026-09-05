

using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Test.Data;

public class MNISTDataLoader : DataLoaderBase
{
    private const string DataDir = @"D:\AI\MNIST";

    public override int ImageScale => 28;
    public override string DatasetName => "MNIST";

    protected override (List<CnnMatrix> trainImages, List<NeuralMatrix> trainLabels,
                        List<CnnMatrix> testImages, List<NeuralMatrix> testLabels)
        LoadBatches(int batchSize, int maxTrainSamples, int maxTestSamples)
    {
        var trainImages = new List<CnnMatrix>();
        var trainLabels = new List<NeuralMatrix>();
        var testImages = new List<CnnMatrix>();
        var testLabels = new List<NeuralMatrix>();

        // Try different possible filename patterns
        string trainImagesPath = FindMNISTFile(DataDir, "train-images", ".idx3-ubyte");
        string trainLabelsPath = FindMNISTFile(DataDir, "train-labels", ".idx1-ubyte");

        if (!File.Exists(trainImagesPath) || !File.Exists(trainLabelsPath))
            throw new FileNotFoundException($"MNIST training files not found in {DataDir}. Please download from: http://yann.lecun.com/exdb/mnist/");

        var (trainImagesData, trainLabelsData) = ReadMNISTBatch(trainImagesPath, trainLabelsPath, maxTrainSamples);
        AddToBatches(trainImagesData, trainLabelsData, batchSize, trainImages, trainLabels);

        string testImagesPath = FindMNISTFile(DataDir, "t10k-images", ".idx3-ubyte");
        string testLabelsPath = FindMNISTFile(DataDir, "t10k-labels", ".idx1-ubyte");

        if (!File.Exists(testImagesPath) || !File.Exists(testLabelsPath))
            throw new FileNotFoundException($"MNIST test files not found in {DataDir}. Please download from: http://yann.lecun.com/exdb/mnist/");

        var (testImagesData, testLabelsData) = ReadMNISTBatch(testImagesPath, testLabelsPath, maxTestSamples);
        AddToBatches(testImagesData, testLabelsData, batchSize, testImages, testLabels);

        Console.WriteLine($"{DatasetName}: Loaded {trainImagesData.Length} training samples in {trainImages.Count} batches");
        Console.WriteLine($"{DatasetName}: Loaded {testImagesData.Length} test samples in {testImages.Count} batches");

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

                // MNIST images are grayscale (28x28), convert to RGB by duplicating the channel
                for (int c = 0; c < Channels; c++)
                {
                    int offset = c * scale * scale;
                    for (int y = 0; y < scale; y++)
                    {
                        for (int x = 0; x < scale; x++)
                        {
                            imgMat[i, c, y, x] = pixels[y * scale + x];
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

    private (float[][] images, int[] labels) ReadMNISTBatch(string imagesPath, string labelsPath, int maxSamples)
    {
        // Read labels
        byte[] labelsData = File.ReadAllBytes(labelsPath);
        int labelOffset = 8; // Skip magic number and count
        int numLabels = labelsData.Length - labelOffset;

        // Read images
        byte[] imagesData = File.ReadAllBytes(imagesPath);
        int imageOffset = 16; // Skip magic number, count, rows, cols
        int imageSize = ImageScale * ImageScale;
        int numImages = (imagesData.Length - imageOffset) / imageSize;

        int samplesToTake = Math.Min(Math.Min(numImages, numLabels), maxSamples);

        var images = new float[samplesToTake][];
        var labels = new int[samplesToTake];

        for (int i = 0; i < samplesToTake; i++)
        {
            labels[i] = labelsData[labelOffset + i];

            float[] img = new float[imageSize];
            int imageStart = imageOffset + i * imageSize;

            for (int j = 0; j < imageSize; j++)
            {
                img[j] = imagesData[imageStart + j] / 255.0f;
            }

            images[i] = img;
        }

        return (images, labels);
    }

    /// <summary>
    /// Finds an MNIST file by trying different naming patterns
    /// </summary>
    private string FindMNISTFile(string directory, string baseName, string extension)
    {
        // Try common naming patterns
        string[] patterns = new[]
        {
            Path.Combine(directory, $"{baseName}{extension}"),           // train-images.idx3-ubyte
            Path.Combine(directory, $"{baseName}-{extension.TrimStart('.')}"), // train-images-idx3-ubyte
            Path.Combine(directory, $"{baseName}.{extension.TrimStart('.')}"), // train-images.idx3-ubyte
            Path.Combine(directory, $"{baseName}"),                      // train-images (if extension is in the file)
        };

        foreach (string pattern in patterns)
        {
            if (File.Exists(pattern))
                return pattern;
        }

        // If none found, try to find any file that contains the base name
        var files = Directory.GetFiles(directory, $"{baseName}*");
        if (files.Length > 0)
            return files[0];

        // Return the most likely path
        return Path.Combine(directory, $"{baseName}{extension}");
    }
}
