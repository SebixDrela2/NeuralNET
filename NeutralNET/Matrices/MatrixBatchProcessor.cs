namespace NeutralNET.Matrices;

public class MatrixBatchProcessor
{
    public IEnumerable<IEnumerable<(MatrixPointer Input, MatrixPointer Output)>> GetBatches(
        Matrix trainingInput,
        Matrix trainingOutput,
        int[] indices,
        int rowCount,
        int batchSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rowCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        if (rowCount == 0)
        {
            yield break;
        }
        
        for (var indicesIndex = 0; indicesIndex < indices.Length; indicesIndex += batchSize)
        {
            yield return EnumerateInBatch(
                indicesIndex, 
                int.Min(indices.Length - indicesIndex, batchSize));
        }

        IEnumerable<(MatrixPointer Input, MatrixPointer Output)> EnumerateInBatch(int indicesOffset, int length)
        {            
            for (var index = 0; index < length; index++)
            {
                var i = indices[index + indicesOffset];

                yield return (
                    Input: trainingInput.GetRowMatrixPointer(i),
                    Output: trainingOutput.GetRowMatrixPointer(i));
            }
        }
    }
}
