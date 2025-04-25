namespace NeutralNET.Matrices;

public class MatrixBatchProcessor
{
    public IEnumerable<IEnumerable<(Memory<float> Input, Memory<float> Output)>> GetBatches(
        IEnumerable<(Memory<float> Input, Memory<float> Output)> rows,
        int rowCount,
        int batchSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rowCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        if (rowCount == 0)
        {
            yield break;
        }

        using var en = rows.GetEnumerator();
        int n;

        for (n = rowCount; n > batchSize; n -= batchSize)
        {
            yield return EnumerateInBatch(batchSize).ToArray();
        }

        yield return EnumerateInBatch(n).ToArray();

        IEnumerable<(Memory<float> Input, Memory<float> Output)> EnumerateInBatch(int count)
        {
            while(count > 0)
            {
                if (!en.MoveNext()) throw new IndexOutOfRangeException();
                --count;
                yield return en.Current;
            }
        }
    }
}
