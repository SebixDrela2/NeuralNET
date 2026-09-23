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
    public override int NumClasses => LettersCount; // 26 uppercase letters A-Z

    public static LetterDataLoader Create() => new();

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

            var imgMat = CnnMatrix.GetOrCreate(currentBatchSize, Channels, scale, scale, readOnly: true);
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

    // private static void PopulateTensorFromPixels(PixelStructRGB pixels, CnnMatrix imgMat, int batchIndex) => PopulateTensorFromPixels(pixels, imgMat, batchIndex, imgMat.Width, imgMat.Height);
    private static void PopulateTensorFromPixels(PixelStructRGB pixels, CnnMatrix imgMat, int batchIndex)
    {
        Debug.Assert(imgMat.Width == GraphicsUtils.Width);
        Debug.Assert(imgMat.Height == GraphicsUtils.Height);
        Debug.Assert(imgMat.Channels == PixelStructRGB.Channels);
        Debug.Assert(batchIndex < imgMat.Batch);

        for (int c = 0; c < PixelStructRGB.Channels; ++c)
        {
            for (int y = 0; y < GraphicsUtils.Height; ++y)
            {
                for (int x = 0; x < GraphicsUtils.Width; ++x)
                {
                    imgMat[batchIndex, c, y, x] = pixels[(y * GraphicsUtils.Height) + x][c];
                }
            }
        }
    }

    private static void PopulateTensorFromPixels(float[] pixels, CnnMatrix imgMat, int batchIndex, int scale)
    {
        var i = 0;

        for (int y = 0; y < scale; y++)
        {
            for (int x = 0; x < scale; x++)
            {
                for (int c = 0; c < Channels; c++, i++)
                {
                    imgMat[batchIndex, c, y, x] = pixels[i];
                }
            }
        }
    }

    /// <summary>
    /// Static helper method for Windows Forms to generate a single sample using the exact
    /// same generation pipeline as the training dataset, returning both the network tensor and UI bitmap.
    /// </summary>
    public static (CnnMatrix ImageTensor, Bitmap DisplayBitmap) GenerateSampleForUI(char targetChar)
    {
        var displayBmp = new Bitmap(GraphicsUtils.Width, GraphicsUtils.Height, PixelFormat.Format32bppArgb);
        var mat = GenerateSampleForUI(targetChar, displayBmp);
        return (mat, displayBmp);
    }

    /// <summary>
    /// Static helper method for Windows Forms to generate a single sample using the exact
    /// same generation pipeline as the training dataset, returning both the network tensor and UI bitmap.
    /// </summary>
    public static CnnMatrix GenerateSampleForUI(char targetChar, Bitmap output)
    {
        var rng = Random.Shared;
        string fontName = FontFamilies[rng.Next(FontFamilies.Length)];
        var style = SupportedStyles[rng.Next(SupportedStyles.Length)];

        var set = GraphicsUtils.GetLettersDataSetRGB(fontName, applyTransformation: true, style: style);

        int targetLabelIndex = char.ToUpper(targetChar) - 'A';
        var sample = set.FirstOrDefault(s => s.Label == targetLabelIndex);

        if (sample.IsEmpty) sample = set[0];

        var imgMat = CnnMatrix.GetOrCreate(1, Channels, GraphicsUtils.Width, GraphicsUtils.Height, readOnly: true);

        PopulateTensorFromPixels(sample, imgMat, 0);

        // var displayBmp = new Bitmap(GraphicsUtils.Width, GraphicsUtils.Height, PixelFormat.Format32bppArgb);
        Debug.Assert(output.Width == GraphicsUtils.Width);
        Debug.Assert(output.Height == GraphicsUtils.Height);
        for (int y = 0; y < GraphicsUtils.Height; ++y)
        {
            for (int x = 0; x < GraphicsUtils.Width; ++x)
            {
                int r = (int)(imgMat[0, 0, y, x] * 0xFF);
                int g = (int)(imgMat[0, 1, y, x] * 0xFF);
                int b = (int)(imgMat[0, 2, y, x] * 0xFF);

                output.SetPixel(x, y, Color.FromArgb(byte.CreateSaturating(r), byte.CreateSaturating(g), byte.CreateSaturating(b)));
            }
        }

        return imgMat;
    }

    public static CnnMatrix LoadInputFromBitmap(Bitmap bmp) => LoadInputFromPixels(GraphicsUtils.GetPixels(bmp), (bmp.Width, bmp.Height));
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

            imgMat ??= CnnMatrix.GetOrCreate(batchLen, Channels, ImageHeight, ImageWidth, readOnly: true);
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
