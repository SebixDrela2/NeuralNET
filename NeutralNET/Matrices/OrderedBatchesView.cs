namespace NeutralNET.Matrices;

public class OrderedBatchesView
{
    private readonly int[] indicies;
    private readonly float[] inRows;
    private readonly float[] outRows;
    
    private readonly int inColSize;
    private readonly int outColSize;

    private readonly int chunkSize;

    private readonly int lastChunkPartSize;
    private readonly int chunksCount;

    public int BatchSize => chunkSize;
    public int BatchCount => chunksCount;
    public int TotalLength => indicies.Length;

    public OrderedBatchesView(
        int[] indicies,
        float[] inRows,
        float[] outRows,
        int inColSize,
        int outColSize,
        int chunkSize)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(inRows.Length, indicies.Length);
        ArgumentOutOfRangeException.ThrowIfNotEqual(outRows.Length, indicies.Length);
        this.indicies = indicies;
        this.inRows = inRows;
        this.outRows = outRows;
        this.inColSize = inColSize;
        this.outColSize = outColSize;
        this.chunkSize = chunkSize;
        (this.chunksCount, this.lastChunkPartSize) = int.DivRem(indicies.Length, chunkSize);
        if (this.lastChunkPartSize != 0) this.chunksCount += 1;
    }
}
