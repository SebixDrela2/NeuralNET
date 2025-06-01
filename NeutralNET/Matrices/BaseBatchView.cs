using static NeutralNET.Matrices.BaseBatchView;

namespace NeutralNET.Matrices;

public abstract class BaseBatchView(int chunkSize, int inputStride, int outputStride)
{
    protected readonly int _chunkSize = chunkSize;

    public int BatchSize => _chunkSize;
    public int InputStride => inputStride;
    public int OutputStride => outputStride;

    protected abstract OrderedBatchView GetCurrentGroup(int offset);
    protected abstract bool MoveNextGroup(ref int offset);
    protected abstract TrainingPair GetCurrent(int offset);
    protected abstract bool MoveNext(ref int offset, int endOffset);


    public GroupEnumerator GetEnumerator() => new(this);

    public struct GroupEnumerator(BaseBatchView self)
    {
        public int Offset = -self.BatchSize;

        public readonly OrderedBatchView Current => self.GetCurrentGroup(Offset);

        public bool MoveNext() => self.MoveNextGroup(ref Offset);
    }

    public struct Enumerator(BaseBatchView self, int offset, int endOffset)
    {
        public int Offset = offset - 1;
        public readonly TrainingPair Current => self.GetCurrent(Offset);

        public bool MoveNext() => self.MoveNext(ref Offset, endOffset);
    }
}

public readonly unsafe struct TrainingPair(float* input, float* output)
{
    public readonly float* Input = input;
    public readonly float* Output = output;

    internal void Deconstruct(out float* input, out float* output)
    {
        input = Input;
        output = Output;
    }
}

public readonly struct OrderedBatchView(BaseBatchView batchesView, int offset, int length)
{
    public int ActualSize => length;
    public BaseBatchView BatchesView => batchesView;

    public Enumerator GetEnumerator() => new(batchesView, offset, offset + length);
}

