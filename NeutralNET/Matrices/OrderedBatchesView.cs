using System.Runtime.CompilerServices;
using static NeutralNET.Matrices.OrderedBatchesView;

namespace NeutralNET.Matrices;

public unsafe class OrderedBatchesView
{
    private readonly int[] _indicies;

    private readonly float* _trainingInput;
    private readonly float* _trainingOutput;

    private readonly int _chunkSize;
    private readonly int _lastChunkPartSize;
    private readonly int _chunksCount;

    public readonly int InputStride;
    public readonly int OutputStride;

    public int BatchSize => _chunkSize;
    public int BatchCount => _chunksCount;
    public int TotalLength => _indicies.Length;

    public OrderedBatchesView(
        int[] indicies,
        NeuralMatrix trainingInput,
        NeuralMatrix trainingOutput,
        int chunkSize)
    {
        _indicies = indicies;
        _trainingInput = trainingInput.Pointer;
        _trainingOutput = trainingOutput.Pointer;

        InputStride = trainingInput.ColumnsStride;
        OutputStride = trainingOutput.ColumnsStride;

        _chunkSize = chunkSize;

        (_chunksCount, _lastChunkPartSize) = int.DivRem(indicies.Length, chunkSize);
        if (_lastChunkPartSize != 0) _chunksCount += 1;
    }

    public GroupEnumerator GetEnumerator() => new(this);

    public struct GroupEnumerator(OrderedBatchesView self)
    {
        public int Offset = -self.BatchSize;

        public readonly OrderedBatchView Current
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return new(self, Offset, int.Min(self.BatchSize, self.TotalLength - Offset));
            }
        }
        
        public bool MoveNext() => (Offset += self.BatchSize) < self.TotalLength;
    }

    public struct Enumerator(OrderedBatchesView batchesView, int offset, int endOffset)
    {
        public int Offset = offset - 1;
        public readonly TrainingPair Current
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                var index = batchesView._indicies[Offset];

                return new(
                    (index * batchesView.InputStride) + batchesView._trainingInput,
                    (index * batchesView.OutputStride) + batchesView._trainingOutput
                );
            }
        }

        public bool MoveNext() => (Offset += 1) < endOffset;
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

public readonly struct OrderedBatchView(OrderedBatchesView batchesView, int offset, int length)
{
    public int ActualSize => length;
    public int InputStride => batchesView.InputStride;
    public int OutputStride => batchesView.OutputStride;

    public Enumerator GetEnumerator() => new(batchesView, offset, offset + length);
}
