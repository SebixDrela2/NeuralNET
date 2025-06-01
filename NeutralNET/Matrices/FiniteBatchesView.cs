namespace NeutralNET.Matrices;

public unsafe class FiniteBatchesView : BaseBatchView
{
    private readonly int[] _indicies;

    private readonly float* _trainingInput;
    private readonly float* _trainingOutput;

    private readonly int _lastChunkPartSize;
    private readonly int _chunksCount;

    public int TotalLength => _indicies.Length;

    public FiniteBatchesView(
        int[] indicies,
        NeuralMatrix trainingInput,
        NeuralMatrix trainingOutput,
        int chunkSize) : base(chunkSize, trainingInput.ColumnsStride, trainingOutput.ColumnsStride)
    {
        _indicies = indicies;
        _trainingInput = trainingInput.Pointer;
        _trainingOutput = trainingOutput.Pointer;

        (_chunksCount, _lastChunkPartSize) = int.DivRem(indicies.Length, chunkSize);

        if (_lastChunkPartSize != 0)
        {
            _chunksCount += 1;
        }
    }

    protected override OrderedBatchView GetCurrentGroup(int offset)
    {
        return new(this, offset, int.Min(BatchSize, TotalLength - offset));
    }

    protected override bool MoveNextGroup(ref int offset) => (offset += BatchSize) < TotalLength;

    protected override TrainingPair GetCurrent(int offset)
    {
        var index = _indicies[offset];

        return new(
            (index * InputStride) + _trainingInput,
            (index * OutputStride) + _trainingOutput
        );
    }

    protected override bool MoveNext(ref int offset, int endOffset) => (offset += 1) < endOffset;
}
