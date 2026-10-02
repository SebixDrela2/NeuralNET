using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.ConstrainedExecution;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using NeutralNET.Matrices;

namespace NeutralNET.Framework.Convolutional;

public unsafe partial class CnnMatrix : CriticalFinalizerObject, IDisposable
{
    public const int Alignment = SIMD.AlignSize;
    private const int ByteAlignment = SIMD.ByteAlignSize;

    public AllocationHandle MemoryHandle;
    public float* Pointer { [MethodImpl(Inline)] get => (float*)MemoryHandle.Pointer; }
    public int Batch { get; set; }
    public int Channels;
    public int Height;
    public int Width;
    public int UnsafeSize;
    public bool ReadOnly;

    public string? DisplayName { get; set; }

    public int StrideW { [MethodImpl(Inline)] get => 1; }
    public int StrideH { [MethodImpl(Inline)] get => Width; }
    public int StrideC { [MethodImpl(Inline)] get => Width * Height; }
    public int StrideN { [MethodImpl(Inline)] get => Width * Height * Channels; }

    private bool _inUse = true;
    private bool _isDisposing = false;

    public static CnnMatrix GetOrCreate(int batch, int channels, int height, int width, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
        => new(batch, channels, height, width, fp, ln);

    private CnnMatrix(int batch, int channels, int height, int width, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0, bool readOnly = false)
    {
        Batch = batch;
        Channels = channels;
        Height = height;
        Width = width;
        UnsafeSize = batch * channels * height * width;

        MemoryHandle = NeuralMemoryPool.Rent<float>(UnsafeSize);

        _inUse = true;
        Clear();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetIndex(int batch, int channel, int y, int x)
    {
        EnsureNotDisposed();
        return (batch * StrideN) + (channel * StrideC) + (y * StrideH) + x;
    }

    public ref float this[int batch, int channel, int y, int x]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            EnsureNotDisposed();
            return ref Pointer[GetIndex(batch, channel, y, x)];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetChannelPointer(int batch, int channel)
    {
        EnsureNotDisposed();
        return Pointer + (batch * StrideN) + (channel * StrideC);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float* GetRowPointer(int batch, int channel, int y)
    {
        EnsureNotDisposed();
        return Pointer + (batch * StrideN) + (channel * StrideC) + (y * StrideH);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Clear()
    {
        EnsureNotDisposed();
        NativeMemory.Clear(Pointer, MemoryHandle.ByteSize);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Fill(float value)
    {
        EnsureNotDisposed();
        new Span<float>(Pointer, UnsafeSize).Fill(value);
    }

    [Conditional("DEBUG")]
    public static void AssertSameSize(CnnMatrix lhs, CnnMatrix rhs, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        AllocationHandle.AssertSameSize(lhs.MemoryHandle, rhs.MemoryHandle, fp, ln);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void CopyFrom(CnnMatrix other, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        EnsureNotDisposed();
        AssertSameSize(this, other, fp, ln);
        NativeMemory.Copy(other.Pointer, Pointer, nuint.Min(MemoryHandle.ByteSize, other.MemoryHandle.ByteSize));
    }



    private readonly Im2ColParallel _im2Col = new();
    public void Im2Col(NeuralMatrix colInput, int kernelH, int kernelW, int stride, int padding)
    {
        EnsureNotDisposed();
        _im2Col.Invoke(this, colInput, kernelH, kernelW, stride, padding);
    }

    private readonly Col2ImParallel _col2Im = new();
    public void Col2Im(NeuralMatrix colGradients)
    {
        EnsureNotDisposed();
        _col2Im.Invoke(this, colGradients);
    }

    public void SetBatch(int batch)
    {
        Batch = batch;
        UnsafeSize = batch * Channels * Height * Width;
        // Note: MemoryHandle.ByteSize stays at the allocation size,
        // which is intentionally larger than UnsafeSize after shrinking.
    }

    [OverloadResolutionPriority(1)]
    public void Dispose([CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        try
        {
            if (_isDisposing)
            {
                throw new InvalidOperationException("ooga");
            }

            _isDisposing = true;
            EnsureNotDisposed();

            _inUse = false;

            MemoryHandle.Take().Dispose();
            GC.SuppressFinalize(this);

            _isDisposing = false;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"BORKED: {ex.Message}");
            throw;
        }
    }

    private void EnsureNotDisposed()
    {
        if (!_inUse)
        {
            throw new NotImplementedException();
        }
    }

    public void Dispose() => Dispose(ln: -1);

    ~CnnMatrix()
    {
        MemoryHandle.Free();
    }
}
