using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Neural.CNN;

public abstract class NeuralRenter
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static NeuralMatrix RentNeural(int rows, int cols, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    => NeuralMatrix.GetOrCreate(rows, cols, fp, ln);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static CnnMatrix RentCnn(int batch, int channels, int h, int w, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
        => CnnMatrix.GetOrCreate(batch, channels, h, w, fp, ln);

    public static CnnMatrix RentCnn(CnnSize cnnSize, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
        => CnnMatrix.GetOrCreate(cnnSize.BatchSize, cnnSize.Channels, cnnSize.Height, cnnSize.Width, fp, ln);
}
