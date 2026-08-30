using NeutralNET.Matrices;
using NeutralNET.Framework.Convolutional;

public static class Cifar10Loader
{
    private const int ImageSize = 32 * 32 * 3; // 3072
    private const int NumClasses = 10;

    public static (List<CnnMatrix> trainImages, List<NeuralMatrix> trainLabels,
                   List<CnnMatrix> testImages, List<NeuralMatrix> testLabels)
        Load(string dataDir, int batchSize = 64, int maxTrainSamples = int.MaxValue, int maxTestSamples = int.MaxValue)
    {
        var trainImages = new List<CnnMatrix>();
        var trainLabels = new List<NeuralMatrix>();
        var testImages = new List<CnnMatrix>();
        var testLabels = new List<NeuralMatrix>();

        int trainSamplesLoaded = 0;
        int testSamplesLoaded = 0;

        // Load training batches
        for (int i = 1; i <= 5; i++)
        {
            if (trainSamplesLoaded >= maxTrainSamples)
                break;

            string path = Path.Combine(dataDir, $"data_batch_{i}.bin");
            if (!File.Exists(path))
                throw new FileNotFoundException($"Training batch not found: {path}");

            var (images, labels) = ReadBatch(path);

            int samplesToTake = Math.Min(images.Length, maxTrainSamples - trainSamplesLoaded);
            if (samplesToTake < images.Length)
            {
                images = images.Take(samplesToTake).ToArray();
                labels = labels.Take(samplesToTake).ToArray();
            }

            AddToBatches(images, labels, batchSize, trainImages, trainLabels);
            trainSamplesLoaded += images.Length;
        }

        // Load test batch
        string testPath = Path.Combine(dataDir, "test_batch.bin");
        if (File.Exists(testPath))
        {
            var (testImgs, testLbls) = ReadBatch(testPath);

            int samplesToTake = Math.Min(testImgs.Length, maxTestSamples);
            if (samplesToTake < testImgs.Length)
            {
                testImgs = testImgs.Take(samplesToTake).ToArray();
                testLbls = testLbls.Take(samplesToTake).ToArray();
            }

            AddToBatches(testImgs, testLbls, batchSize, testImages, testLabels);
            testSamplesLoaded += testImgs.Length;
        }

        Console.WriteLine($"Loaded {trainSamplesLoaded} training samples in {trainImages.Count} batches");
        Console.WriteLine($"Loaded {testSamplesLoaded} test samples in {testImages.Count} batches");

        return (trainImages, trainLabels, testImages, testLabels);
    }

    private static (float[][] images, int[] labels) ReadBatch(string path)
    {
        byte[] data = File.ReadAllBytes(path);
        int numSamples = data.Length / (ImageSize + 1);

        var images = new float[numSamples][];
        var labels = new int[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            int offset = i * (ImageSize + 1);
            labels[i] = data[offset];

            float[] img = new float[ImageSize];
            for (int j = 0; j < ImageSize; j++)
                img[j] = data[offset + 1 + j] / 255.0f;
            images[i] = img;
        }

        return (images, labels);
    }

    private static void AddToBatches(float[][] images, int[] labels, int batchSize,
                                     List<CnnMatrix> outImages, List<NeuralMatrix> outLabels)
    {
        int numSamples = images.Length;
        for (int start = 0; start < numSamples; start += batchSize)
        {
            int end = Math.Min(start + batchSize, numSamples);
            int currentBatchSize = end - start;

            var imgMat = CnnMatrix.GetOrCreate(currentBatchSize, 3, 32, 32, readOnly:true);
            var lblMat = NeuralMatrix.GetOrCreate(currentBatchSize, 10);

            for (int i = 0; i < currentBatchSize; i++)
            {
                int idx = start + i;
                float[] pixels = images[idx];

                // Copy pixels using the indexer
                for (int c = 0; c < 3; c++)
                {
                    int offset = c * 1024; // 32*32
                    for (int y = 0; y < 32; y++)
                    {
                        for (int x = 0; x < 32; x++)
                        {
                            imgMat[i, c, y, x] = pixels[offset + y * 32 + x];
                        }
                    }
                }

                // One-hot label
                int label = labels[idx];
                lblMat.Set(i, label, 1.0f);
            }

            outImages.Add(imgMat);
            outLabels.Add(lblMat);
        }
    }
}
