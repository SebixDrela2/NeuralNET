using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Test.Data;

public class NeuralDataset
{
    public List<CnnMatrix> TrainImages { get; set; } = [];
    public List<NeuralMatrix> TrainLabels { get; set; } = [];
    public int[] TrainActualLabels { get; set; } = [];
    public List<CnnMatrix> TestImages { get; set; } = [];
    public List<NeuralMatrix> TestLabels { get; set; } = [];
    public int[] TestActualLabels { get; set; } = [];

    public int TrainSampleCount => TrainImages?.Sum(b => b.Batch) ?? 0;
    public int TestSampleCount => TestImages?.Sum(b => b.Batch) ?? 0;

    public void Dispose()
    {
        //? TrainImages.ClearAndDispose();
        TrainLabels.ClearAndDispose();
        TestImages.ClearAndDispose();
        TestLabels.ClearAndDispose();
    }
}
