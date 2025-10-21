using NeutralNET.Matrices;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NeutralNET.Unmanaged;

public readonly unsafe struct NeuralVector(float* pointer, int columns, int stride)
{
    public readonly float* Pointer = pointer;
    public readonly int Columns = columns;
    public readonly int Stride = stride;
    public readonly uint UsedByteSize { [MethodImpl(Inline)] get => unchecked((uint)(sizeof(float) * Columns)); }

    public Span<float> Span { [MethodImpl(Inline)] get => new(Pointer, Columns); }

    [MethodImpl(Inline)]
    public static NeuralVector operator ++(NeuralVector row) => row + 1;

    [MethodImpl(Inline)]
    public static NeuralVector operator --(NeuralVector row) => row - 1;

    [MethodImpl(Inline)]
    public static NeuralVector operator +(NeuralVector row, int count) => new(row.Pointer + row.Stride * count, row.Columns, row.Stride);

    [MethodImpl(Inline)]
    public static NeuralVector operator -(NeuralVector row, int count) => row + -count;

    [MethodImpl(Inline)]
    public Vector512<float> LoadVectorUnaligned(int index) => SIMD.LoadUnaligned(Pointer + index);

    [MethodImpl(Inline)]
    public Vector512<float> LoadVectorAligned(int index)
    {
        AssertAligned();
        return SIMD.LoadAligned(Pointer + index);
    }

    [MethodImpl(Inline)] public void CopyTo(MatrixElement other) => CopyTo(other.Pointer);
    [MethodImpl(Inline)] public void CopyTo(float* ptr) => NativeMemory.Copy(Pointer, ptr, UsedByteSize);

    [Conditional("DEBUG"), MethodImpl(Inline)]
    private void AssertAligned() => Debug.Assert((SIMD.ByteAlignMask & (nuint)Pointer) is 0);
}
