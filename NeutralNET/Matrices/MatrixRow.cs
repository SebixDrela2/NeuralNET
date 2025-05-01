using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace NeutralNET.Matrices;

public unsafe readonly struct MatrixRow(
    float* pointer, 
    int rows, 
    int columns,
    int stride)
{
    public readonly float* Pointer = pointer;

    public readonly int Rows = rows;
    public readonly int Columns = columns;
    public readonly int Stride = stride;
    
    public Span<float> Span
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => new(Pointer, Columns);
    }
    //public ref float this[int index] => ref Pointer[index * Columns];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static MatrixRow operator++ (MatrixRow row) => row + 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static MatrixRow operator--(MatrixRow row) => row - 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static MatrixRow operator+(MatrixRow row, int count) => new(row.Pointer + (row.Stride * count), row.Rows, row.Columns, row.Stride);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static MatrixRow operator-(MatrixRow row, int count) => row + (-count);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector256<float> LoadVectorUnaligned(int index)
    {       
        return Vector256.LoadUnsafe(ref *Pointer, (nuint)index);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector256<float> LoadVectorAligned(int index)
    {
        AssertAligned();

        return Vector256.LoadAligned(Pointer + index);
    }

    [Conditional("DEBUG")]
    private void AssertAligned()
    {
        UIntPtr ptr = (UIntPtr)Pointer;

        var end = (ptr & MatrixElement.UnalignedBits) == 0;

        Debug.Assert(end);
    }

    [Conditional("DEBUG")]
    private void TraceUnaligned()
    {
        UIntPtr ptr = (UIntPtr)Pointer;

        var end = (ptr & MatrixElement.UnalignedBits) != 0;

        if (end)
        {
            Console.WriteLine("Unneccesary aligned.");
        }
    }
}
