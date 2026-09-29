using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.Versioning;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Utils;

namespace NeutralNET.Test.Data;

[SupportedOSPlatform("windows6.1")]
public class LetterDataLoader : DataLoaderBase, IDataLoader<LetterDataLoader>
{
    public const DataSourceType DataSourceType = DataSourceType.Letters;

    private static readonly string[] FontFamilies =
    [
        "Consolas", "Arial", "Times New Roman", "Georgia", "Verdana", "Tahoma",
        //"Consolas", "Courier New", "Comic Sans MS", "Impact", "Trebuchet MS",
        //"Palatino Linotype", "Segoe UI", "Lucida Console", "Garamond", "Century Gothic"
    ];

    private static FontStyle[] SupportedStyles => [FontStyle.Regular];

    public const int LettersCount = 'Z' - 'A' + 1;

    public override int ImageWidth => GraphicsUtils.Width;
    public override int ImageHeight => GraphicsUtils.Height;

    public override string DatasetName => "LetterData";
    public override int NumClasses => LettersCount; 

    private static bool _normalizationStatsReady;
    private static readonly object _statsLock = new();

    public static LetterDataLoader Create()
    {
        EnsureNormalizationStats();
        return new();
    }

    private static void EnsureNormalizationStats()
    {
        if (_normalizationStatsReady) return;

        lock (_statsLock)
        {
            if (_normalizationStatsReady) return;

            var prevEnabled = InputNormalization.Enabled;
            InputNormalization.Enabled = false;

            try
            {
                const int SampleCount = 2000;
                var rng = Random.Shared;

                var transforms = new GraphicsUtils.CharTransformation[SampleCount];
                for (int i = 0; i < SampleCount; i++)
                {
                    transforms[i] = new(
                        FontFamilies[rng.Next(FontFamilies.Length)],
                        GraphicsUtils.ImageTransformation.CreateRandom(rng),
                        Color.GetRandomColors());
                }

                var samples = GraphicsUtils.GetLettersDataSetRGB(
                    transforms,
                    GraphicsUtils.DefaultLetters,
                    randomCharOrder: true);

                InputNormalization.ComputeStats(samples);

                Console.WriteLine(
                    $"[InputNormalization] mean=({InputNormalization.MeanR:F4}, " +
                    $"{InputNormalization.MeanG:F4}, {InputNormalization.MeanB:F4})  " +
                    $"invStd=({InputNormalization.InvStdR:F4}, " +
                    $"{InputNormalization.InvStdG:F4}, {InputNormalization.InvStdB:F4})");
            }
            finally
            {
                InputNormalization.Enabled = prevEnabled;
                _normalizationStatsReady = true;
            }
        }
    }

    protected override NeuralDataSet LoadBatches(int batchSize, (int Train, int Test) samples)
    {
        var trainSet = CreateTrainSet(DataSetType.Train, batchSize, samples.Train);
        var testSet = CreateTrainSet(DataSetType.Test, batchSize, samples.Test);

        return new(trainSet, testSet);
    }

    protected override void UpdateBatches(int batchSize, (int Train, int Test) samples, NeuralDataSet dataSet)
    {
        UpdateTrainSet(DataSetType.Train, batchSize, samples.Train, dataSet.Train);
        UpdateTrainSet(DataSetType.Test, batchSize, samples.Test, dataSet.Test);
    }

    protected override void AddToBatches(float[][] images, int[] labels, int batchSize, List<CnnMatrix> outImages, List<NeuralMatrix> outLabels)
    {
        Debug.Assert(ImageWidth == ImageHeight);
        var scale = ImageWidth;

        int numSamples = images.Length;

        for (int start = 0; start < numSamples; start += batchSize)
        {
            int end = Math.Min(start + batchSize, numSamples);
            int currentBatchSize = end - start;

            if (currentBatchSize <= 0) break;

            var imgMat = CnnMatrix.GetOrCreate(currentBatchSize, Channels, scale, scale);
            var lblMat = NeuralMatrix.GetOrCreate(currentBatchSize, NumClasses);

            for (int i = 0; i < currentBatchSize; i++)
            {
                int idx = start + i;
                float[] pixels = images[idx];

                PopulateTensorFromPixels(pixels, imgMat, i, scale);

                int label = labels[idx];
                lblMat.Set(i, label, 1.0f);
            }

            outImages.Add(imgMat);
            outLabels.Add(lblMat);
        }
    }

    private static void PopulateTensorFromPixels(PixelStructRGB pixels, CnnMatrix imgMat, int batchIndex)
    {
        Debug.Assert(imgMat.Width == GraphicsUtils.Width);
        Debug.Assert(imgMat.Height == GraphicsUtils.Height);
        Debug.Assert(imgMat.Channels == PixelStructRGB.Channels);
        Debug.Assert(batchIndex < imgMat.Batch);

        int width = GraphicsUtils.Width;
        int height = GraphicsUtils.Height;

        for (int c = 0; c < PixelStructRGB.Channels; ++c)
        {
            for (int y = 0; y < height; ++y)
            {
                int rowOffset = y * width;   // BUG FIX: was `y * height`
                for (int x = 0; x < width; ++x)
                {
                    imgMat[batchIndex, c, y, x] = pixels[rowOffset + x][c];
                }
            }
        }
    }

    private static void PopulateTensorFromPixels(float[] pixels, CnnMatrix imgMat, int batchIndex, int scale)
    {
        var i = 0;
        bool normalize = InputNormalization.Enabled;

        for (int y = 0; y < scale; y++)
        {
            for (int x = 0; x < scale; x++)
            {
                float r = pixels[i++];
                float g = pixels[i++];
                float b = pixels[i++];

                if (normalize)
                {
                    (r, g, b) = InputNormalization.Normalize(r, g, b);
                }

                imgMat[batchIndex, 0, y, x] = r;
                imgMat[batchIndex, 1, y, x] = g;
                imgMat[batchIndex, 2, y, x] = b;
            }
        }
    }

    public static (CnnMatrix ImageTensor, Bitmap DisplayBitmap) GenerateSampleForUI(char targetChar)
    {
        var displayBmp = new Bitmap(GraphicsUtils.Width, GraphicsUtils.Height, PixelFormat.Format32bppArgb);
        var mat = GenerateSampleForUI(targetChar, displayBmp);
        return (mat, displayBmp);
    }

    public static CnnMatrix GenerateSampleForUI(char targetChar, Bitmap output)
    {
        var rng = Random.Shared;
        string fontName = FontFamilies[rng.Next(FontFamilies.Length)];
        var style = SupportedStyles[rng.Next(SupportedStyles.Length)];

        var set = GraphicsUtils.GetLettersDataSetRGB(fontName, applyTransformation: true, style: style);

        int targetLabelIndex = char.ToUpper(targetChar) - 'A';
        var sample = set.FirstOrDefault(s => s.Label == targetLabelIndex);

        if (sample.IsEmpty) sample = set[0];

        var imgMat = CnnMatrix.GetOrCreate(1, Channels, GraphicsUtils.Width, GraphicsUtils.Height);

        PopulateTensorFromPixels(sample, imgMat, 0);

        Debug.Assert(output.Width == GraphicsUtils.Width);
        Debug.Assert(output.Height == GraphicsUtils.Height);

        bool denormalize = InputNormalization.Enabled;

        for (int y = 0; y < GraphicsUtils.Height; ++y)
        {
            for (int x = 0; x < GraphicsUtils.Width; ++x)
            {
                float r = imgMat[0, 0, y, x];
                float g = imgMat[0, 1, y, x];
                float b = imgMat[0, 2, y, x];

                if (denormalize)
                {
                    (r, g, b) = InputNormalization.Denormalize(r, g, b);
                }

                int rb = (int)(Math.Clamp(r, 0f, 1f) * 0xFF);
                int gb = (int)(Math.Clamp(g, 0f, 1f) * 0xFF);
                int bb = (int)(Math.Clamp(b, 0f, 1f) * 0xFF);

                output.SetPixel(x, y, Color.FromArgb((byte)rb, (byte)gb, (byte)bb));
            }
        }

        return imgMat;
    }

    public static CnnMatrix LoadInputFromBitmap(Bitmap bmp)
        => LoadInputFromPixels(GraphicsUtils.GetPixels(bmp), (bmp.Width, bmp.Height));

    public static CnnMatrix LoadInputFromPixels(PixelStructRGB pixels, (int Width, int Height) size)
    {
        var imgMat = CnnMatrix.GetOrCreate(1, Channels, size.Width, size.Height);
        PopulateTensorFromPixels(pixels, imgMat, 0);
        return imgMat;
    }

    public override NeuralTrainSet CreateTrainSet(DataSetType dataSetType, int batchSize, int maxSamples)
    {
        var allSamples = new PixelStructRGB[maxSamples];
        foreach (ref var item in allSamples.AsSpan()) item = new PixelStructRGB(default, GraphicsUtils.PixelCount);

        var batchCount = (maxSamples + (batchSize - 1)) / batchSize;
        var batchImages = new CnnMatrix[batchCount];
        var batchLabels = new NeuralMatrix[batchCount];

        NeuralTrainSet output = new()
        {
            Images = allSamples,
            ImagesData = batchImages,
            LabelsData = batchLabels,
        };
        UpdateTrainSet(dataSetType, batchSize, maxSamples, output);
        return output;
    }

    public override void UpdateTrainSet(DataSetType dataSetType, int batchSize, int maxSamples, NeuralTrainSet output)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSamples);

        bool isTrain = dataSetType == DataSetType.Train;
        var rng = Random.Shared;

        if (SupportedStyles is not [FontStyle.Regular]) throw new InvalidOperationException();
        var rngProps = new GraphicsUtils.CharTransformation[maxSamples];

        for (int i = 0; i < maxSamples; i++)
        {
            rngProps[i] = new(
                FontFamilies[rng.Next(FontFamilies.Length)],
                isTrain ? GraphicsUtils.ImageTransformation.CreateRandom(rng) : GraphicsUtils.ImageTransformation.None,
                Color.GetRandomColors()
            );
        }

        var allSamples = output.Images;
        var batchImages = output.ImagesData;
        var batchLabels = output.LabelsData;

        GraphicsUtils.GetLettersDataSetRGB(rngProps, allSamples, GraphicsUtils.DefaultLetters);

        int[] indices = [.. Enumerable.Range(0, maxSamples)];
        rng.Shuffle(indices);

        for (var (pos, batchIndex) = (0, 0); pos < maxSamples; ++batchIndex)
        {
            int batchEnd = int.Min(pos + batchSize, maxSamples);
            int batchLen = batchEnd - pos;

            ref var imgMat = ref batchImages[batchIndex];
            ref var lblMat = ref batchLabels[batchIndex];

            imgMat ??= CnnMatrix.GetOrCreate(batchLen, Channels, ImageHeight, ImageWidth);
            lblMat ??= NeuralMatrix.GetOrCreate(batchLen, NumClasses);

            for (int j = 0; pos < batchEnd; ++j, ++pos)
            {
                ref readonly var item = ref allSamples[indices[pos]];

                PopulateTensorFromPixels(item, imgMat, j);

                lblMat[j].Clear();
                lblMat[j, item.Label] = 1;
            }
        }

        Console.WriteLine($"{DatasetName}: Loaded {output.Images.Length} {dataSetType} samples with randomized transform diversity.");
    }

    static DataSourceType IDataLoader<LetterDataLoader>.DataSourceType => DataSourceType;
}
