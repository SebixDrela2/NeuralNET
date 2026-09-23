using System.Diagnostics.CodeAnalysis;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using NeutralNET.Utils;

namespace NeutralNET.Test.Data;

public sealed class NeuralTrainSet : IDisposable
{
    private CnnMatrix[]? _imagesData;
    private NeuralMatrix[]? _labelsData;

    [AllowNull]
    public required PixelStructRGB[] Images { get => field ??= []; set; }

    [AllowNull]
    public required CnnMatrix[] ImagesData { get => _imagesData ??= []; set => Exchange(ref _imagesData, value)?.DisposeEach(); }
    [AllowNull]
    public required NeuralMatrix[] LabelsData { get => _labelsData ??= []; set => Exchange(ref _labelsData, value)?.DisposeEach(); }

    public void Dispose() => (ImagesData, LabelsData) = (null, null);
}

public sealed class NeuralDataSet(NeuralTrainSet train, NeuralTrainSet test) : IDisposable
{
    public NeuralTrainSet Train { get; } = train;
    public NeuralTrainSet Test { get; } = test;

    // public List<CnnMatrix> TrainImages { get; set; } = [];
    // public List<NeuralMatrix> TrainLabels { get; set; } = [];
    // public int[] TrainActualLabels { get; set; } = [];
    // public List<CnnMatrix> TestImages { get; set; } = [];
    // public List<NeuralMatrix> TestLabels { get; set; } = [];
    // public int[] TestActualLabels { get; set; } = [];

    // public int TrainSampleCount => TrainImages?.Sum(b => b.Batch) ?? 0;
    // public int TestSampleCount => TestImages?.Sum(b => b.Batch) ?? 0;

    public void Dispose()
    {
        Train.Dispose();
        Test.Dispose();
        // TrainImages.ClearAndDispose();
        // TrainLabels.ClearAndDispose();
        // TestImages.ClearAndDispose();
        // TestLabels.ClearAndDispose();
    }
}
