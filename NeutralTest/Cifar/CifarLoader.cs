using NeutralNET.Matrices;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Stuff;

public static class Cifar10Loader
{
    private const int Scale = 16;
    private const int ImageSize = Scale * Scale * 3; // 3072
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

    private static readonly string[] _fontNames =
    ["Arial", "Times New Roman", "Georgia", "Verdana", "Tahoma"];


    private static (List<CnnMatrix> images, List<NeuralMatrix> labels) LoadAndFlattenDataSet(
        DataSetType dataSetType,
        int batchSize,
        int maxSamples)
    {
        var batchImages = new List<CnnMatrix>();
        var batchLabels = new List<NeuralMatrix>();
        int samplesLoaded = 0;
        var fontNames = _fontNames.ToArray();
        int batchCount = dataSetType == DataSetType.Train ? 5 : 1;

        for (int i = 1; i <= batchCount; i++)
        {
            if (samplesLoaded >= maxSamples)
            {
                break;
            }

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

        Console.WriteLine($"Loaded {samplesLoaded} {dataSetType} samples in {batchImages.Count} batches");

        var flatImages = FlattenBatchesToImages(batchImages);
        var flatLabels = FlattenBatchesToLabels(batchLabels);

        foreach (var img in batchImages) img.Dispose();
        foreach (var lbl in batchLabels) lbl.Dispose();

        Console.WriteLine($"Flattened to {flatImages.Count} individual {dataSetType} images");

        return (flatImages, flatLabels);
    }

    public static (
        List<CnnMatrix> trainImages,
        List<NeuralMatrix> trainLabels,
        List<CnnMatrix> testImages,
        List<NeuralMatrix> testLabels) GenerateDigiDigi(
        int batchSize = 64,
        int maxTrainSamples = int.MaxValue,
        int maxTestSamples = int.MaxValue)
    {
        var (trainImages, trainLabels) = LoadAndFlattenDataSet(
            DataSetType.Train,
            batchSize,
            maxTrainSamples);

        var (testImages, testLabels) = LoadAndFlattenDataSet(
            DataSetType.Test,
            batchSize,
            maxTestSamples);

        return (trainImages, trainLabels, testImages, testLabels);
    }

    public static (List<CnnMatrix> trainImages, List<NeuralMatrix> trainLabels) GetNewTrainData(int batchSize, int maxTrainSamples)
    {
        var (trainImages, trainLabels) = LoadAndFlattenDataSet(
            DataSetType.Train,
            batchSize,
            maxTrainSamples);

        return (trainImages, trainLabels);
    }

    public static List<CnnMatrix> FlattenBatchesToImages(List<CnnMatrix> batches)
    {
        var images = new List<CnnMatrix>();
        foreach (var batch in batches)
        {
            for (int i = 0; i < batch.Batch; i++)
            {
                var single = CnnMatrix.GetOrCreate(1, 3, Scale, Scale);
                for (int c = 0; c < 3; c++)
                    for (int y = 0; y < Scale; y++)
                        for (int x = 0; x < Scale; x++)
                            single[0, c, y, x] = batch[i, c, y, x];
                images.Add(single);
            }
        }
        return images;
    }

    // ✅ NEW: Flatten batches to individual labels
    public static List<NeuralMatrix> FlattenBatchesToLabels(List<NeuralMatrix> batches)
    {
        var labels = new List<NeuralMatrix>();
        foreach (var batch in batches)
        {
            for (int i = 0; i < batch.Rows; i++)
            {
                var single = NeuralMatrix.GetOrCreate(1, 10);
                for (int j = 0; j < 10; j++)
                {
                    single.At(0, j) = batch.At(i, j);
                }
                labels.Add(single);
            }
        }
        return labels;
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

            var imgMat = CnnMatrix.GetOrCreate(currentBatchSize, 3, Scale, Scale, readOnly: true);
            var lblMat = NeuralMatrix.GetOrCreate(currentBatchSize, 10);

            for (int i = 0; i < currentBatchSize; i++)
            {
                int idx = start + i;
                float[] pixels = images[idx];

                // Copy pixels using the indexer
                for (int c = 0; c < 3; c++)
                {
                    int offset = c * Scale * Scale;
                    for (int y = 0; y < Scale; y++)
                    {
                        for (int x = 0; x < Scale; x++)
                        {
                            imgMat[i, c, y, x] = pixels[offset + y * Scale + x];
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

    private enum DataSetType
    {
        Train,
        Test
    }
}
