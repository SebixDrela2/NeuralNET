using System.Collections;

namespace NeutralNET.Test.Enumerator;

public abstract class BatchEnumerator<TImage, TLabel> : IEnumerator<(TImage image, TLabel label, int actualLabel)>
{
    protected int _currentIndex = -1;
    protected List<TImage> _images;
    protected List<TLabel> _labels;
    protected int[] _actualLabels;
    protected int _currentBatchIndex = 0;
    protected int _currentSampleIndex = 0;

    public abstract bool MoveNext();
    public abstract (TImage image, TLabel label, int actualLabel) Current { get; }
    public abstract void Reset();
    public abstract void Dispose();

    object IEnumerator.Current => Current;
}
