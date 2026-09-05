
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Test.Cifar;

public class CifarDataset
{
    public List<CnnMatrix> TrainImages { get; set; }
    public List<NeuralMatrix> TrainLabels { get; set; }
    public int[] TrainActualLabels { get; set; }
    public List<CnnMatrix> TestImages { get; set; }
    public List<NeuralMatrix> TestLabels { get; set; }
    public int[] TestActualLabels { get; set; }

    public int TrainSampleCount => TrainImages.Sum(b => b.Batch);
    public int TestSampleCount => TestImages.Sum(b => b.Batch);

    public void Dispose()
    {
        foreach (var img in TrainImages) img?.Dispose();
        foreach (var lbl in TrainLabels) lbl?.Dispose();
        foreach (var img in TestImages) img?.Dispose();
        foreach (var lbl in TestLabels) lbl?.Dispose();
    }
}
