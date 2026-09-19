using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.Versioning;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Utils;
using Tensorflow;

namespace NeutralNET.Test.Data;

[SupportedOSPlatform("windows6.1")]
public class LetterDataLoader : DataLoaderBase
{
    private static readonly string[] FontFamilies =
    [
        "Consolas", "Arial", "Times New Roman", "Georgia", "Verdana", "Tahoma",
        //"Consolas", "Courier New", "Comic Sans MS", "Impact", "Trebuchet MS",
        //"Palatino Linotype", "Segoe UI", "Lucida Console", "Garamond", "Century Gothic"
    ];

    private static FontStyle[] SupportedStyles => [FontStyle.Regular];

    public const int LettersCount = 'Z' - 'A' + 1;

    public override int ImageScale => GraphicsUtils.Width;
    public override string DatasetName => "LetterData";
    public override int NumClasses => LettersCount; // 26 uppercase letters A-Z

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

    /// <summary>
    /// Unified shared helper method to populate a CnnMatrix slice from raw flat pixel data,
    /// ensuring exact parity between training batch creation and Windows Forms UI generation.
    /// </summary>
    ///
    private static void PopulateTensorFromPixels(PixelStructRGB pixels, CnnMatrix imgMat, int batchIndex, int scale)
    {
        for (int y = 0; y < scale; y++)
        {
            for (int x = 0; x < scale; x++)
            {
                for (int c = 0; c < Channels; c++)
                {
                    imgMat[batchIndex, c, y, x] = pixels.Pixels[y * scale + x][c];
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
        var rng = Random.Shared;
        string fontName = FontFamilies[rng.Next(FontFamilies.Length)];
        var style = SupportedStyles[rng.Next(SupportedStyles.Length)];

        var set = GraphicsUtils.GetLettersDataSetRGB(fontName, applyTransformation: true, style: style);

        int targetLabelIndex = char.ToUpper(targetChar) - 'A';
        var sample = set.FirstOrDefault(s => s.Label == targetLabelIndex);

        if (sample.IsEmpty) sample = set[0];

        Debug.Assert(GraphicsUtils.Width == GraphicsUtils.Height);
        int scale = GraphicsUtils.Width;

        var imgMat = CnnMatrix.GetOrCreate(1, Channels, GraphicsUtils.Width, GraphicsUtils.Height, readOnly: true);

        PopulateTensorFromPixels(sample.Flat.ToArray(), imgMat, 0, scale);

        var displayBmp = new Bitmap(GraphicsUtils.Width, GraphicsUtils.Height, PixelFormat.Format32bppArgb);
        for (int y = 0; y < GraphicsUtils.Height; ++y)
        {
            for (int x = 0; x < GraphicsUtils.Width; ++x)
            {
                int r = (int)(imgMat[0, 0, y, x] * 0xFF);
                int g = (int)(imgMat[0, 1, y, x] * 0xFF);
                int b = (int)(imgMat[0, 2, y, x] * 0xFF);

                displayBmp.SetPixel(x, y, Color.FromArgb(byte.CreateSaturating(r), byte.CreateSaturating(g), byte.CreateSaturating(b)));
            }
        }

        return (imgMat, displayBmp);
    }

    private (List<CnnMatrix> images, List<NeuralMatrix> labels) LoadFlattenedDataSet(DataSetType dataSetType, int batchSize, int maxSamples)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSamples);

        bool isTrain = dataSetType == DataSetType.Train;
        var rng = Random.Shared;

        var fontTemplates = new List<(string FontName, FontStyle Style)>();

        foreach (var fontName in FontFamilies)
        {
            var stylesToRender = isTrain ? SupportedStyles : [FontStyle.Regular];
            foreach (var style in stylesToRender)
            {
                fontTemplates.Add((fontName, style));
            }
        }

        if (fontTemplates.Count == 0) fontTemplates.Add(("Arial", FontStyle.Regular));

        var allSamples = new List<PixelStructRGB>(maxSamples);

        int n = maxSamples - allSamples.Count;
        while (true)
        {
            var template = fontTemplates[rng.Next(fontTemplates.Count)];

            var set = GraphicsUtils.GetLettersDataSetRGB(template.FontName, applyTransformation: isTrain, style: template.Style);
            rng.Shuffle(set);

            if (set.Length > n)
            {
                allSamples.AddRange(set.Take(n));
                break;
            }

            allSamples.AddRange(set);
            n -= set.Length;
        }

        Debug.Assert(allSamples.Count == maxSamples);
        var selectedData = allSamples.Take(maxSamples).ToArray(); // ?

        int[] indices = Enumerable.Range(0, selectedData.Length).ToArray();
        rng.Shuffle(indices);

        var labelArray = new int[selectedData.Length];
        var imageArray = new float[selectedData.Length][];

        for (int i = 0; i < indices.Length; i++) // shuffled order on already shuffled data?
        {
            var selected = selectedData[indices[i]];

            labelArray[i] = selected.Label;
            imageArray[i] = selected.Flat.ToArray();
        }

        var batchImages = new List<CnnMatrix>();
        var batchLabels = new List<NeuralMatrix>();

        AddToBatches(imageArray, labelArray, batchSize, batchImages, batchLabels);

        Console.WriteLine($"{DatasetName}: Loaded {selectedData.Length} {dataSetType} samples with randomized transform diversity.");

        return (batchImages, batchLabels);
    }

    private enum DataSetType
    {
        Train,
        Test
    }
}
