using System.Runtime.InteropServices;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Convolutional;

/// <summary>
/// Safe 4D tensor (Batch, Channels, Height, Width) using managed arrays.
/// No unsafe code, no pointers.
/// </summary>
public class CnnMatrix : IDisposable
{
    private float[] _data;
    private bool _disposed;

    public int Batch { get; }
    public int Channels { get; }
    public int Height { get; }
    public int Width { get; }
    public bool ReadOnly { get; }
    public int AllocatedLength { get; }

    // Strides for NCHW layout
    public int StrideW => 1;
    public int StrideH => Width;
    public int StrideC => Width * Height;
    public int StrideN => Width * Height * Channels;

    public CnnMatrix(int batch, int channels, int height, int width, bool readOnly = false)
    {
        Batch = batch;
        Channels = channels;
        Height = height;
        Width = width;
        AllocatedLength = batch * channels * height * width;
        _data = new float[AllocatedLength];
        ReadOnly = readOnly;
        Clear();
    }

    /// <summary>
    /// Gets a pointer to the start of a specific (batch, channel, y, x) location.
    /// Returns the index into the flat array.
    /// </summary>
    public int GetIndex(int batch, int channel, int y, int x)
        => (batch * StrideN) + (channel * StrideC) + (y * StrideH) + (x * StrideW);

    /// <summary>
    /// Gets a reference to a specific element.
    /// </summary>
    public ref float this[int batch, int channel, int y, int x]
        => ref _data[GetIndex(batch, channel, y, x)];

    /// <summary>
    /// Gets a span for a whole channel.
    /// </summary>
    public Span<float> GetChannelSpan(int batch, int channel)
    {
        int start = (batch * StrideN) + (channel * StrideC);
        return _data.AsSpan(start, Height * Width);
    }


    /// <summary>
    /// Gets a span for a specific row within a channel.
    /// </summary>
    public Span<float> GetRowSpan(int batch, int channel, int y)
    {
        int start = (batch * StrideN) + (channel * StrideC) + (y * StrideH);
        return _data.AsSpan(start, Width);
    }

    public void Clear()
    {
        Array.Clear(_data, 0, _data.Length);
    }

    public void Fill(float value)
    {
        Array.Fill(_data, value);
    }

    public void CopyFrom(CnnMatrix other)
    {
        if (other.AllocatedLength != AllocatedLength)
            throw new ArgumentException("Size mismatch");
        Array.Copy(other._data, _data, AllocatedLength);
    }

    public float[] ToArray() => (float[])_data.Clone();

    /// <summary>
    /// Extracts image patches into a 2D NeuralMatrix.
    /// </summary>
    /// <summary>
    /// Extracts image patches into a 2D NeuralMatrix.
    /// Output rows = (OutH * OutW), Output columns = (Channels * KernelH * KernelW)
    /// </summary>
    public NeuralMatrix Im2Col(int kernelH, int kernelW, int stride, int padding)
    {
        int paddedH = Height + 2 * padding;
        int paddedW = Width + 2 * padding;
        int outH = (paddedH - kernelH) / stride + 1;
        int outW = (paddedW - kernelW) / stride + 1;
        int patchSize = Channels * kernelH * kernelW;

        var colMatrix = new NeuralMatrix(outH * outW, patchSize);
        using var padded = new CnnMatrix(Batch, Channels, paddedH, paddedW);
        padded.Clear();

        // Copy with padding using direct indexing
        for (int b = 0; b < Batch; b++)
        {
            for (int c = 0; c < Channels; c++)
            {
                for (int y = 0; y < Height; y++)
                {
                    for (int x = 0; x < Width; x++)
                    {
                        padded[b, c, y + padding, x + padding] = this[b, c, y, x];
                    }
                }
            }
        }

        // Extract patches using direct indexing
        for (int b = 0; b < Batch; b++)
        {
            for (int c = 0; c < Channels; c++)
            {
                int channelOffset = c * kernelH * kernelW;
                for (int oh = 0; oh < outH; oh++)
                {
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int startY = oh * stride;
                        int startX = ow * stride;
                        int patchRow = (oh * outW + ow);

                        for (int ky = 0; ky < kernelH; ky++)
                        {
                            for (int kx = 0; kx < kernelW; kx++)
                            {
                                int colIdx = channelOffset + ky * kernelW + kx;
                                colMatrix.At(patchRow, colIdx) = padded[b, c, startY + ky, startX + kx];
                            }
                        }
                    }
                }
            }
        }

        return colMatrix;
    }

    /// <summary>
    /// Scatters gradients from a 2D NeuralMatrix back into this tensor.
    /// </summary>
    public void Col2Im(NeuralMatrix colGradients, int kernelH, int kernelW, int stride, int padding, float scale = 1.0f)
    {
        int paddedH = Height + 2 * padding;
        int paddedW = Width + 2 * padding;
        int outH = (paddedH - kernelH) / stride + 1;
        int outW = (paddedW - kernelW) / stride + 1;
        int patchSize = Channels * kernelH * kernelW;

        using var paddedGrad = new CnnMatrix(Batch, Channels, paddedH, paddedW);
        paddedGrad.Clear();

        for (int b = 0; b < Batch; b++)
        {
            for (int oh = 0; oh < outH; oh++)
            {
                for (int ow = 0; ow < outW; ow++)
                {
                    int startY = oh * stride;
                    int startX = ow * stride;
                    int patchRow = (oh * outW + ow);

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

        // Copy back from padded (excluding padding)
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
        if (ReadOnly)
        {
            throw new Exception();
        }

        if (!_disposed)
        {
            _data = null!;
            _disposed = true;
        }
    }
}
