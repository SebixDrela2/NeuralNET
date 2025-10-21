using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NeutralNET.Matrices;

public readonly unsafe struct MatrixElement(float* pointer, int length)
{
    public readonly float* Pointer = pointer;

    public readonly int Count = length;
    public Span<float> Span => new(Pointer, Count);
    public ref float Reference => ref *Pointer;

    public nuint ByteLenght => sizeof(float) * (nuint)Count;

    public ref float this[int index] => ref Pointer[index];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void CopyTo(MatrixElement other)
    {
        NativeMemory.Copy(Pointer, other.Pointer, ByteLenght);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector512<float> LoadVectorAligned(int index)
    {
        AssertAligned(index);
        return Vector512.LoadAligned(Pointer + index);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector512<float> LoadVectorUnaligned(int index) => Vector512.LoadUnsafe(ref Reference, (nuint)index);

    [Conditional("DEBUG"), MethodImpl(Inline)] private void AssertAligned() => Debug.Assert(SIMD.IsAligned(Pointer));
    [Conditional("DEBUG"), MethodImpl(Inline)] private void AssertAligned(int offset) => Debug.Assert(SIMD.IsAligned(Pointer + offset));
}
