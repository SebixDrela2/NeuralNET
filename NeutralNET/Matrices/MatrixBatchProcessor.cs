namespace NeutralNET.Matrices;

public class MatrixBatchProcessor
{
    public IEnumerable<(Matrix InputBatch, Matrix OutputBatch)> GetBatches(
        Matrix inputMatrix,
        Matrix outputMatrix,
        int batchSize)
    {
        if (inputMatrix.Rows != outputMatrix.Rows)
        {
            throw new ArgumentException("Input and output matrices must have same number of rows");
        }

        if (batchSize <= 0)
        {
            throw new ArgumentException("Batch size must be positive", nameof(batchSize));
        }

        for (int startRow = 0; startRow < inputMatrix.Rows; startRow += batchSize)
        {
            yield return (
                inputMatrix.BatchView(startRow, batchSize),
                outputMatrix.BatchView(startRow, batchSize)
            );
        }
    }
}
