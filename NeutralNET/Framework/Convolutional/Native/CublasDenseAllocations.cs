namespace NeutralNET.Framework.Convolutional.Native;

public record class CublasDenseAllocations(
    CublasContext Layer) : IDisposable
{
    public void Dispose()
    {
        Layer.Dispose();
    }
}
