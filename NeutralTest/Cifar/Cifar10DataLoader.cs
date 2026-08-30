using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

public static class Cifar10DataLoader
{
    public static (List<CnnMatrix> images, List<NeuralMatrix> labels, int[] actualLabels)
        LoadBatches(string dataDir, int batchSize = 10, int maxSamples = 100)
    {
        var (trainImages, trainLabels, _, _) = Cifar10Loader.Load(
            dataDir: dataDir,
            batchSize: batchSize,
            maxTrainSamples: maxSamples,
            maxTestSamples: 0
        );

        // Read actual labels
        int totalSamples = trainLabels.Sum(l => l.Rows);
        int[] actualLabels = new int[totalSamples];
        int offset = 0;
        foreach (var lbl in trainLabels)
        {
            for (int i = 0; i < lbl.Rows; i++)
                actualLabels[offset + i] = ArgMax(lbl.GetRowSpan(i));
            offset += lbl.Rows;
        }

        return (trainImages, trainLabels, actualLabels);
    }

    private static int ArgMax(Span<float> row)
    {
        int maxIdx = 0;
        float maxVal = row[0];
        for (int i = 1; i < row.Length; i++)
        {
            if (row[i] > maxVal)
            {
                maxVal = row[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
