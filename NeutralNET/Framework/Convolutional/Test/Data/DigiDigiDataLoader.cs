using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Utils;

namespace NeutralNET.Test.Data;

// public class DigiDigiDataLoader : DataLoaderBase
// {
//     private static readonly string[] _fontNames =
//         ["Arial", "Times New Roman", "Georgia", "Verdana", "Tahoma"];

//     public override int ImageWidth => 16;
//     public override int ImageHeight => 16;
//     public override string DatasetName => "DigiDigi";

//     protected override (List<CnnMatrix> trainImages, List<NeuralMatrix> trainLabels,
//                         List<CnnMatrix> testImages, List<NeuralMatrix> testLabels)
//         LoadBatches(int batchSize, int maxTrainSamples, int maxTestSamples)
//     {
//         var (trainImages, trainLabels) = LoadFlattenedDataSet(DataSetType.Train, batchSize, maxTrainSamples);
//         var (testImages, testLabels) = LoadFlattenedDataSet(DataSetType.Test, batchSize, maxTestSamples);

//         return (trainImages, trainLabels, testImages, testLabels);
//     }

//     protected override void AddToBatches(float[][] images, int[] labels, int batchSize,
//                                          List<CnnMatrix> outImages, List<NeuralMatrix> outLabels)
//     {
//         int scale = ImageScale;
//         int numSamples = images.Length;

//         for (int start = 0; start < numSamples; start += batchSize)
//         {
//             int end = Math.Min(start + batchSize, numSamples);
//             int currentBatchSize = end - start;

//             var imgMat = CnnMatrix.GetOrCreate(currentBatchSize, Channels, scale, scale, readOnly: true);
//             var lblMat = NeuralMatrix.GetOrCreate(currentBatchSize, NumClasses);

//             for (int i = 0; i < currentBatchSize; i++)
//             {
//                 int idx = start + i;
//                 float[] pixels = images[idx];

//                 for (int c = 0; c < Channels; c++)
//                 {
//                     int offset = c * scale * scale;
//                     for (int y = 0; y < scale; y++)
//                     {
//                         for (int x = 0; x < scale; x++)
//                         {
//                             imgMat[i, c, y, x] = pixels[offset + y * scale + x];
//                         }
//                     }
//                 }

//                 int label = labels[idx];
//                 lblMat.Set(i, label, 1.0f);
//             }

//             outImages.Add(imgMat);
//             outLabels.Add(lblMat);
//         }
//     }

//     private (List<CnnMatrix> images, List<NeuralMatrix> labels) LoadFlattenedDataSet(DataSetType dataSetType, int batchSize, int maxSamples)
//     {
//         var batchImages = new List<CnnMatrix>();
//         var batchLabels = new List<NeuralMatrix>();

//         // Generate synthetic font dataset iterations cleanly without skipping
//         int passes = dataSetType == DataSetType.Train ? 5 : 1;
//         bool applyTransform = dataSetType == DataSetType.Train;

//         var allSamples = new List<PixelStructRGB>();

//         for (int i = 0; i < passes; i++)
//         {
//             foreach (var font in _fontNames)
//             {
//                 var fontData = GraphicsUtils.GetDigitsDataSetRGB(font, applyTransformation: applyTransform);
//                 allSamples.AddRange(fontData);
//             }
//         }

//         // Shuffle all collected samples
//         var sampleArray = allSamples.ToArray();
//         Random.Shared.Shuffle(sampleArray);

//         // Take only up to maxSamples requested
//         var selectedData = sampleArray.Take(maxSamples).ToArray();

//         var labelArray = selectedData.Select(x => x.Label).ToArray();
//         var imageArray = selectedData.Select(x => x.Flat.ToArray()).ToArray();

//         AddToBatches(imageArray, labelArray, batchSize, batchImages, batchLabels);

//         Console.WriteLine($"{DatasetName}: Loaded {selectedData.Length} {dataSetType} samples in {batchImages.Count} batches");

//         var flatImages = FlattenBatchesToImages(batchImages);
//         var flatLabels = FlattenBatchesToLabels(batchLabels);

//         foreach (var img in batchImages) img?.Dispose();
//         foreach (var lbl in batchLabels) lbl?.Dispose();

//         return (flatImages, flatLabels);
//     }

//     private List<CnnMatrix> FlattenBatchesToImages(List<CnnMatrix> batches)
//     {
//         int scale = ImageScale;
//         var images = new List<CnnMatrix>();

//         foreach (var batch in batches)
//         {
//             for (int i = 0; i < batch.Batch; i++)
//             {
//                 var single = CnnMatrix.GetOrCreate(1, Channels, scale, scale);
//                 for (int c = 0; c < Channels; c++)
//                     for (int y = 0; y < scale; y++)
//                         for (int x = 0; x < scale; x++)
//                             single[0, c, y, x] = batch[i, c, y, x];
//                 images.Add(single);
//             }
//         }
//         return images;
//     }

//     private List<NeuralMatrix> FlattenBatchesToLabels(List<NeuralMatrix> batches)
//     {
//         var labels = new List<NeuralMatrix>();

//         foreach (var batch in batches)
//         {
//             for (int i = 0; i < batch.Rows; i++)
//             {
//                 var single = NeuralMatrix.GetOrCreate(1, NumClasses);
//                 for (int j = 0; j < NumClasses; j++)
//                     single.At(0, j) = batch.At(i, j);
//                 labels.Add(single);
//             }
//         }
//         return labels;
//     }

//     private enum DataSetType
//     {
//         Train,
//         Test
//     }
// }
