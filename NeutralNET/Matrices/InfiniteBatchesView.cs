using NeutralNET.Models;
using NeutralNET.Utils;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NeutralNET.Matrices;

internal unsafe class InfiniteBatchesView : BaseBatchView
{
    private int _lastOffset;
    private int _lastOffsetGroup;

    private float* _lastValues;

    public IDynamicModel Model;

    public InfiniteBatchesView(IDynamicModel model, int chunkSize)
        : this(model, chunkSize, MatrixUtils.GetStride(2), MatrixUtils.GetStride(1)) { }

    public InfiniteBatchesView(IDynamicModel model, int chunkSize, int inputStride, int outputStride)
        : base(chunkSize, inputStride, outputStride)
    {
        Model = model;

        var count = inputStride + outputStride;

        var byteCount = (nuint)(count * sizeof(float) * BatchSize);
        var alignment = (nuint)(Vector256<float>.Count * sizeof(float));

        _lastValues = (float*)NativeMemory.AlignedAlloc(byteCount, alignment);       
    }

    public override void Dispose()
    {
        if (_lastValues is not null)
        {
            NativeMemory.Free(_lastValues);

            _lastValues = null;
        }

        base.Dispose();
    }

    protected override TrainingPair GetCurrent(int offset)
    {
        if (offset != _lastOffset)
        {
            throw new InvalidOperationException();
        }

        var relativeOffset = offset % BatchSize;

        var lastValues = _lastValues + (relativeOffset * RowStride);
        var lastValuesOutput = lastValues + InputStride;

        return new TrainingPair(lastValues, lastValuesOutput);
    }

    protected override OrderedBatchView GetCurrentGroup(int offset)
    {
        if (offset != _lastOffsetGroup)
        {
            throw new InvalidOperationException();
        }

        return new OrderedBatchView(this, offset, BatchSize);
    }

    protected override bool MoveNext(ref int offset, int endOffset)
    {
        if ((offset += 1) >= endOffset)
        {
            return false;
        }

        _lastOffset = offset;

        return true;
    }

    protected override bool MoveNextGroup(ref int offset)
    {
        offset += BatchSize;

        _lastOffsetGroup = offset;

        var ptr = _lastValues;

        for (var i = 0; i < BatchSize; ++i, ptr += RowStride)
        {
            var scaledInput = Model.ScaleDown(Random.Shared.NextSingle() * 10);
            var scaledInput2 = Model.ScaleDown(Random.Shared.NextSingle() * 10);

            ptr[0] = scaledInput;
            ptr[1] = scaledInput2;

            var scaledOutput = Model.ScaleDown(Model.PrepareFunction(ptr[0], ptr[1]));

            ptr[InputStride] = scaledOutput;
        }

        return true;
    }
}
