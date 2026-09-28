namespace NeutralNET.Framework.Convolutional.Native;

public record class CublasReverseDenseAllocations(
    CublasContext DWeights,
    CublasContext GradInput) : IDisposable
{
    public void Dispose()
    {
        DWeights.Dispose();
        GradInput.Dispose();
    }
}
