namespace NeutralNET.Test.Enumerator;

public interface IDataBatchProvider<TImage, TLabel>
{
    IEnumerable<(TImage image, TLabel label, int actualLabel)> GetTrainingBatches(int batchSize);
    IEnumerable<(TImage image, TLabel label, int actualLabel)> GetTestBatches(int batchSize);
    int TotalTrainingSamples { get; }
    int TotalTestSamples { get; }
}
