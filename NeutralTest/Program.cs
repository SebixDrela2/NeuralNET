using NeutralNET.Activation;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;
using NeutralNET.Framework.Connected.Optimizers;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Test.Data;

namespace NeutralTest;

internal class Program
{
    static void Main() => RunCnnNetwork();

    public static void RunCnnNetwork()
    {
        var loader = DataLoaderFactory.Create(DataSourceType.MNIST);
        var dataSet = loader.LoadCompleteDataset(
           batchSize: 64,
           maxTrainSamples: 1000,
           maxTestSamples: 1000
       );

        var cnnConfig = new CnnArchitectureConfig
        {
            ConvLayers =
    [
        new() {
            KernelHeight = 3,
            KernelWidth = 3,
            Filters = 32,                    
            Stride = 1,
            Padding = 1,
            Activation = ActivationType.LeakyReLU,
            UseMaxPool = true,
            PoolSize = 2
        },
        new() {
            KernelHeight = 3,
            KernelWidth = 3,
            Filters = 64,                    
            Stride = 1,
            Padding = 1,
            Activation = ActivationType.LeakyReLU,
            UseMaxPool = true,
            PoolSize = 2
        },
        new() {
            KernelHeight = 3,
            KernelWidth = 3,
            Filters = 128,                   
            Stride = 1,
            Padding = 1,
            Activation = ActivationType.LeakyReLU,
            UseMaxPool = true,
            PoolSize = 2
        }
    ],
            DenseArchitecture = [256, 128, 10],          
            DenseHiddenActivation = ActivationType.LeakyReLU,
            OutputActivation = ActivationType.Softmax,
            OptimizerConfig = new CnnOptimizerConfig
            {
                OptimizerType = CnnOptimizerType.Adam,
                LearningRate = 0.001f,               
                WeightDecay = 0.0001f,               
                Beta1 = 0.9f,
                Beta2 = 0.999f,
                Epsilon = 1e-8f
            }
        };

        var denseConfig = new NeuralNetworkConfig
        {
            LearningRate = 0.001f,
            WeightDecay = 0.0001f,                  
            BatchSize = 64,                         
            Epochs = 1,
            DropoutRate = 0.25f,                    
            WithShuffle = true,
            OptimizerType = OptimizerType.Adam,
            Model = null
        };

        using var network = new CnnBuilder<Architecture>()
            .WithCnnConfig(cnnConfig)
            .WithDenseConfig(denseConfig)
            .WithInputSize(loader.ImageScale, loader.ImageScale, 3)
            .Build();
        var validator = new CnnValidator();
        DiagnoseNetwork(network, dataSet);
        TrainAndRenderTable(network, validator, dataSet, loader.DatasetName);

        Console.WriteLine("\n=== FINAL EVALUATION ===");
        var finalResult = validator.Validate(network, dataSet.TestImages, dataSet.TestLabels);
        validator.PrintResults(finalResult);

        foreach (var img in dataSet.TrainImages) img.Dispose();
        foreach (var lbl in dataSet.TrainLabels) lbl.Dispose();
        foreach (var img in dataSet.TestImages) img.Dispose();
        foreach (var lbl in dataSet.TestLabels) lbl.Dispose();
    }

    private static void DiagnoseNetwork(CnnNetwork<Architecture> network, NeuralDataset dataSet)
    {
        Console.WriteLine("\n=== NETWORK DIAGNOSTIC ===\n");

        // 1. Check input data
        var firstImage = dataSet.TrainImages[0];
        Console.WriteLine($"Input shape: {firstImage.Batch}x{firstImage.Channels}x{firstImage.Height}x{firstImage.Width}");

        float min = float.MaxValue, max = float.MinValue, sum = 0;
        int count = 0;
        for (int b = 0; b < firstImage.Batch; b++)
            for (int c = 0; c < firstImage.Channels; c++)
                for (int y = 0; y < firstImage.Height; y++)
                    for (int x = 0; x < firstImage.Width; x++)
                    {
                        float v = firstImage[b, c, y, x];
                        if (v < min) min = v;
                        if (v > max) max = v;
                        sum += v;
                        count++;
                    }
        Console.WriteLine($"Pixel values: Min={min:F4}, Max={max:F4}, Avg={sum / count:F4}");
        Console.WriteLine($"Expected: Min=0, Max=1, Avg~0.1-0.2\n");

        // 2. Check labels
        var firstLabel = dataSet.TrainLabels[0];
        Console.WriteLine($"Label shape: {firstLabel.Rows}x{firstLabel.UsedColumns}");
        Console.Write("First label row: ");
        for (int j = 0; j < firstLabel.UsedColumns; j++)
            Console.Write($"{firstLabel.At(0, j):F0} ");
        Console.WriteLine("\n");

        // 3. Forward pass debug - get raw logits and softmax output
        Console.WriteLine("=== FORWARD PASS DEBUG ===");
        var pred = network.Forward(firstImage);
        Console.WriteLine($"Output shape: {pred.Rows}x{pred.UsedColumns}");

        Console.Write("Softmax output (first row): ");
        float sumProbs = 0;
        for (int j = 0; j < 10; j++)
        {
            float v = pred.At(0, j);
            sumProbs += v;
            Console.Write($"{v:F4} ");
        }
        Console.WriteLine($"\nSum of probabilities: {sumProbs:F6} (should be ~1.0)");

        if (sumProbs < 0.5f)
            Console.WriteLine("❌ SOFTMAX IS BROKEN - all outputs are zero!");
        pred.Dispose();

        // 4. Train one batch and check gradients
        Console.WriteLine("\n=== GRADIENT DEBUG ===");
        float loss = network.TrainBatch(firstImage, firstLabel, 0.001f);
        Console.WriteLine($"Loss after one batch: {loss:F4} (should be ~2.3, not 14.5)");

        if (loss > 10f)
            Console.WriteLine("❌ LOSS TOO HIGH - gradients are not working!");
    }

    private static void TrainAndRenderTable(
    CnnNetwork<Architecture> network,
    CnnValidator validator,
    NeuralDataset dataSet,
    string datasetName)
    {
        var learningRate = 0.001f;
        var maxEpochs = 2000;
        var earlyStopPatience = 3000;
        var targetAccuracy = 0.99f;
        var bestAccuracy = 0f;
        var epochsSinceBest = 0;

        // Get test data as single batches for validation
        var testImages = dataSet.TestImages;
        var testLabels = dataSet.TestLabels;

        for (int epoch = 0; epoch < maxEpochs; epoch++)
        {
            var totalLoss = 0f;

            for (int batchIdx = 0; batchIdx < dataSet.TrainImages.Count; batchIdx++)
            {
                float loss = network.TrainBatch(dataSet.TrainImages[batchIdx], dataSet.TrainLabels[batchIdx], learningRate);
                totalLoss += loss;
            }

            var avgLoss = totalLoss / dataSet.TrainImages.Count;

            //if ((epoch & ((1 << 1) - 1)) == 0)
            {
                var result = validator.Validate(network, testImages, testLabels);
                float accuracy = result.Accuracy;
                Console.Write("\e[H");
                Console.WriteLine($"╔══════════════╤══════════════════╤════════════════════╤═════════════════════════════╗");
                Console.WriteLine($"║  Epoch {epoch + 1,5} │ Loss: {avgLoss,9:F6}  │  Accuracy: {accuracy,7:P2} │  Best: {bestAccuracy,7:P2}              ║");
                Console.WriteLine($"╠═══════╤══════╧══════════════════╧════════════════════╧══════════════╤══════════════╣");
                Console.WriteLine($"║       │     0     1     2     3     4     5     6     7     8     9 │ Pred  Actual ║");
                Console.WriteLine($"╠═══════╪═════════════════════════════════════════════════════════════╪══════════════╣");

                var numSamples = Math.Min(10, testImages.Count);
                for (int i = 0; i < numSamples; i++)
                {
                    var pred = network.Forward(testImages[i]);
                    var probs = new float[10];

                    for (var j = 0; j < 10; j++)
                    {
                        probs[j] = pred.At(0, j);
                    }

                    var predicted = ArgMax(probs);
                    var actual = GetActualLabel(testLabels[i]);

                    Console.Write($"║ {i,2}    │");
                    for (int j = 0; j < 10; j++)
                    {
                        var v = probs[j];
                        var s = FmtPogression(v);
                        if (v < 0.1f) Console.Write($" \e[2m{s}\e[22m");
                        else Console.Write($" {s}");
                    }

                    Console.WriteLine($" │  {(predicted == actual ? AsGreen(predicted.ToString()) : AsRed(predicted.ToString())),2}      {actual,2}   ║");

                    pred.Dispose();
                }

                Console.WriteLine($"╚═══════╧═════════════════════════════════════════════════════════════╧══════════════╝");
                Console.WriteLine();
                Console.WriteLine($"Best accuracy: {bestAccuracy:P2}  |  Epochs since best: {epochsSinceBest}");

                if (accuracy > bestAccuracy)
                {
                    bestAccuracy = accuracy;
                    epochsSinceBest = 0;
                    Console.WriteLine($"*** NEW BEST! ***");
                }
                else
                {
                    epochsSinceBest++;
                }

                if (accuracy >= targetAccuracy)
                {
                    Console.WriteLine($"\n🎯 Target accuracy {targetAccuracy:P2} reached! Stopping early at epoch {epoch + 1}");
                    break;
                }

                if (epochsSinceBest >= earlyStopPatience && bestAccuracy > 0.5f)
                {
                    Console.WriteLine($"\n⏹️ No improvement for {earlyStopPatience} epochs. Stopping early at epoch {epoch + 1}");
                    Console.WriteLine($"Best accuracy: {bestAccuracy:P2}");
                    break;
                }
            }
        }
    }

    private static string FmtPogression(float x)
    {
        const int ColWidth = 5;
        const char ChZero = ' ';
        const char ChMax = '\u2588';

        switch (x)
        {
            case 0: return $"\e[48;2;74;74;76m{x,5:f3}\e[49m";
            case 1: return $"\e[7;48;2;74;74;76m{x,5:f3}\e[49;27m";
            case <= 0: return $"\e[48;2;110;74;76m{x,5:f2}\e[49m";
            case >= 1: return $"\e[7;48;2;74;74;76m{x,5:f3}\e[49;27m";
        }
        Span<char> xs = stackalloc char[ColWidth];
        xs.Fill(ChZero);

        var scaled = x * ColWidth;
        var maxEnd = int.Clamp((int)scaled, 0, ColWidth);
        xs[..maxEnd].Fill(ChMax);

        if (maxEnd != ColWidth)
        {
            int frame = int.Clamp((int)((8 * (scaled - maxEnd)) + 0.5f), 0, 7);
            xs[maxEnd] = (char)(ChMax + (7 - frame));
        }

        return $"\e[48;2;74;74;76m{xs}\e[49m";
    }

    private static int ArgMax(float[] array)
    {
        int maxIdx = 0;
        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > array[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    private static int GetActualLabel(NeuralMatrix labelMatrix)
    {
        for (int i = 0; i < 10; i++)
        {
            if (labelMatrix.At(0, i) > 0.5f) return i;
        }
        return -1;
    }

    private static string AsGreen(string x) => $"\e[38;2;124;179;66m{x}\e[39m";
    private static string AsRed(string x) => $"\e[38;2;230;74;25m{x}\e[39m";
}
