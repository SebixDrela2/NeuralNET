using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NeutralNET.Matrices;

public readonly unsafe struct MatrixPointer(float* pointer, int length)
{
    public const int UnalignedBits = 31;

    public readonly float* Pointer = pointer;

    public readonly int Count = length;
    public Span<float> Span => new(Pointer, Count);
    public ref float Reference => ref *Pointer;

    public nuint ByteLenght => sizeof(float) * (nuint)Count;

    public ref float this[int index] => ref Pointer[index];
    
    public void CopyTo(MatrixPointer other)
    {
        NativeMemory.Copy(Pointer, other.Pointer, ByteLenght);
    }

    public Vector256<float> LoadVectorAligned(int index)
    {
        AssertAligned();
        
        return Vector256.LoadAligned(Pointer + index);
    }

    public Vector256<float> LoadVectorUnAligned(int index) => Vector256.LoadUnsafe(ref Reference, (nuint)index);

    [Conditional("DEBUG")]
    private void AssertAligned()
    {
        UIntPtr ptr = (UIntPtr)Pointer;

        var end = (ptr & UnalignedBits) == 0;

        Debug.Assert(end);
    }
}
