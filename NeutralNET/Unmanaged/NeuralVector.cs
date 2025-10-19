﻿using NeutralNET.Matrices;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NeutralNET.Unmanaged;

public unsafe readonly struct NeuralVector(
    float* pointer,
    int columns,
    int stride)
{
    public readonly float* Pointer = pointer;
    public readonly int Columns = columns;
    public readonly int Stride = stride;

    public Span<float> Span
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => new(Pointer, Columns);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static NeuralVector operator ++(NeuralVector row) => row + 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static NeuralVector operator --(NeuralVector row) => row - 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static NeuralVector operator +(NeuralVector row, int count) => new(row.Pointer + row.Stride * count, row.Columns, row.Stride);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static NeuralVector operator -(NeuralVector row, int count) => row + -count;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector512<float> LoadVectorUnaligned(int index)
    {
        return Vector512.LoadUnsafe(ref *Pointer, (nuint)index);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Vector512<float> LoadVectorAligned(int index)
    {
        AssertAligned();

        return Vector512.LoadAligned(Pointer + index);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void CopyTo(MatrixElement other)
    {
        var byteLength = sizeof(float) * Columns;

        NativeMemory.Copy(Pointer, other.Pointer, (uint)byteLength);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void CopyTo(float* ptr)
    {
        var byteLength = sizeof(float) * Columns;

        NativeMemory.Copy(Pointer, ptr, (uint)byteLength);
    }

    [Conditional("DEBUG")]
    private void AssertAligned()
    {
        nuint ptr = (nuint)Pointer;

        var end = (ptr & MatrixElement.UnalignedBits) == 0;

        Debug.Assert(end);
    }

    [Conditional("DEBUG")]
    private void TraceUnaligned()
    {
        nuint ptr = (nuint)Pointer;

        var end = (ptr & MatrixElement.UnalignedBits) != 0;

        if (end)
        {
            Console.WriteLine("Unneccesary aligned.");
        }
    }
}
