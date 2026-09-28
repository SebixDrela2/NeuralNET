namespace NeutralNET.Framework.Convolutional.Native;

public record class CublasConvAllocations(
    CublasContext GradPatchMat,
    CublasContext DWeights,
    CublasContext Convolution) : IDisposable
{
    public void Dispose()
    {
        GradPatchMat.Dispose();
        DWeights.Dispose();
        Convolution.Dispose();
    }
}
