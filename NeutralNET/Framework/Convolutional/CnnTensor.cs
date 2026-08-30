using NeutralNET.Matrices;

namespace NeutralNET.Framework.Convolutional;

/// <summary>
/// 4D tensor (Batch, Channels, Height, Width) with array pooling for performance.
/// </summary>
public class CnnMatrix : IDisposable
{
    private readonly float[] _data;
    private readonly int _allocatedLength;

    private static readonly Stack<CnnMatrix> _pool = [];

    public static CnnMatrix GetOrCreate(int batch, int channels, int height, int width, bool readOnly = false)
    {
        if (!_pool.TryPop(out var item))
        {
            item = new CnnMatrix(batch, channels, height, width, readOnly);

            return item;
        }

        item.Resize(batch, channels, height, width);

        return item;
    }

    private static readonly int CommonAllocatedLength = 163840;

    public int Batch;
    public int Channels;
    public int Height;
    public int Width;
    public bool ReadOnly;

    public int StrideW = 1;
    public int StrideH;
    public int StrideC;
    public int StrideN;

    public int UnsafeSize;

    private CnnMatrix(int batch, int channels, int height, int width, bool readOnly = false)
    {
        Batch = batch;
        Channels = channels;
        Height = height;
        Width = width;
        _allocatedLength = CommonAllocatedLength;
        UnsafeSize = batch * channels * height * width;
        ReadOnly = readOnly;
        _data = new float[_allocatedLength];
        StrideH = Width;
        StrideC = Width * Height;
        StrideN = Width * Height * Channels;
        Clear();
    }

    public void Resize(int batch, int channels, int height, int width)
    {
        Batch = batch;
        Channels = channels;
        Height = height;
        Width = width;

        var allocatedLength = batch * channels * height * width;

        if (_data.Length < allocatedLength)
        {
            throw new Exception();
        }

        UnsafeSize = batch * channels * height * width;
        StrideH = Width;
        StrideC = Width * Height;
        StrideN = Width * Height * Channels;

        Clear();
    }

    [MethodImpl(Inline)]
    public int GetIndex(int batch, int channel, int y, int x)
        => (batch * StrideN) + (channel * StrideC) + (y * StrideH) + (x * StrideW);

    public ref float this[int batch, int channel, int y, int x]
    {
        [MethodImpl(Inline)]
        get => ref _data[GetIndex(batch, channel, y, x)];
    }

    [MethodImpl(Inline)]
    public Span<float> GetChannelSpan(int batch, int channel)
    {
        int start = (batch * StrideN) + (channel * StrideC);

        return _data.AsSpan(start, Height * Width);
    }

    public void Clear()
    {
        Array.Clear(_data, 0, UnsafeSize);
    }

    [MethodImpl(Inline)]
    public void CopyFrom(CnnMatrix other)
    {
        if (other._allocatedLength != _allocatedLength)
        {
            throw new ArgumentException("Size mismatch");
        }

        Array.Copy(other._data, 0, _data, 0, UnsafeSize);
    }

    public NeuralMatrix Im2Col(int kernelH, int kernelW, int stride, int padding)
    {
        int paddedH = Height + 2 * padding;
        int paddedW = Width + 2 * padding;
        int outH = (paddedH - kernelH) / stride + 1;
        int outW = (paddedW - kernelW) / stride + 1;
        int patchSize = Channels * kernelH * kernelW;

        var colMatrix = NeuralMatrix.GetOrCreate(Batch * outH * outW, patchSize);
        using var padded = GetOrCreate(Batch, Channels, paddedH, paddedW);
        padded.Clear();

        for (int b = 0; b < Batch; b++)
        {
            for (int c = 0; c < Channels; c++)
            {
                var srcSpan = GetChannelSpan(b, c);
                var dstSpan = padded.GetChannelSpan(b, c);
                int dstOffset = padding * padded.Width + padding;

                for (int y = 0; y < Height; y++)
                {
                    int srcRowOffset = y * Width;
                    int dstRowOffset = dstOffset + (y + padding) * padded.Width;

                    for (int x = 0; x < Width; x++)
                    {
                        dstSpan[dstRowOffset + x] = srcSpan[srcRowOffset + x];
                    }
                }
            }
        }

        for (int b = 0; b < Batch; b++)
        {
            for (int c = 0; c < Channels; c++)
            {
                var paddedSpan = padded.GetChannelSpan(b, c);
                int channelOffset = c * kernelH * kernelW;

                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int startY = oh * stride;
                        int startX = ow * stride;
                        int patchRow = (b * outH + oh) * outW + ow;

                        int kyOffset = 0;
                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            int rowStart = (startY + ky) * padded.Width + startX;
                            for (int kx = 0; kx < kernelW; kx++)
                            {
                                int colIdx = channelOffset + kyOffset + kx;
                                int paddedIdx = rowStart + kx;
                                colMatrix.At(patchRow, colIdx) = paddedSpan[paddedIdx];
                            }
                            kyOffset += kernelW;
                        }
                    }
                }
            }
        }

        return colMatrix;
    }

    public void Col2Im(NeuralMatrix colGradients, int kernelH, int kernelW, int stride, int padding, float scale = 1.0f)
    {
        int paddedH = Height + 2 * padding;
        int paddedW = Width + 2 * padding;
        int outH = (paddedH - kernelH) / stride + 1;
        int outW = (paddedW - kernelW) / stride + 1;
        int patchSize = Channels * kernelH * kernelW;

        using var paddedGrad = GetOrCreate(Batch, Channels, paddedH, paddedW);
        paddedGrad.Clear();

        for (int b = 0; b < Batch; b++)
        {
            for (int oh = 0; oh < outH; oh++)
            {
                for (int ow = 0; ow < outW; ow++)
                {
                    int startY = oh * stride;
                    int startX = ow * stride;
                    int patchRow = (b * outH + oh) * outW + ow;

                    for (int c = 0; c < Channels; c++)
                    {
                        int channelOffset = c * kernelH * kernelW;
                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            for (int kx = 0; kx < kernelW; kx++)
                            {
                                int colIdx = channelOffset + ky * kernelW + kx;
                                float val = colGradients.At(patchRow, colIdx) * scale;
                                paddedGrad[b, c, startY + ky, startX + kx] += val;
                            }
                        }
                    }
                }
            }
        }

        for (int b = 0; b < Batch; b++)
        {
            for (int c = 0; c < Channels; c++)
            {
                for (int y = 0; y < Height; y++)
                {
                    for (int x = 0; x < Width; x++)
                    {
                        this[b, c, y, x] = paddedGrad[b, c, y + padding, x + padding];
                    }
                }
            }
        }
    }

    public void Dispose()
    {
        _pool.Push(this);
    }
}
