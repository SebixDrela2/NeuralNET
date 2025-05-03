using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NeutralNET.Matrices;

public readonly unsafe struct MatrixElement(float* pointer, int length)
{
    public const int UnalignedBits = 31;

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
    public Vector256<float> LoadVectorAligned(int index)
    {
        AssertAligned();
        
        return Vector256.LoadAligned(Pointer + index);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector256<float> LoadVectorUnaligned(int index) => Vector256.LoadUnsafe(ref Reference, (nuint)index);

    [Conditional("DEBUG")]
    private void AssertAligned()
    {
        UIntPtr ptr = (UIntPtr)Pointer;

        var end = (ptr & UnalignedBits) == 0;

        Debug.Assert(end);
    }
}
