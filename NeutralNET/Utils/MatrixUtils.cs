using NeutralNET.Matrices;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NeutralNET.Utils;

public static class MatrixUtils
{
    public static readonly uint[][] StrideMaskLookup = MakeStrideMaskLookup();
    public static int GetStride(int columns) => (columns + SIMD.AlignMask) & ~SIMD.AlignMask;

    // TODO: ensure this is not modified
    public static uint[] GetStrideMask(int columns) => StrideMaskLookup[GetLeftoverSize(columns)];
    private static int GetLeftoverSize(int columns) => columns & SIMD.AlignMask;
    private static int GetPadSize(int columns) => SIMD.AlignSize - GetLeftoverSize(columns);

    static uint[][] MakeStrideMaskLookup()
    {
        var result = new uint[SIMD.AlignSize][];

        Span<uint> prev = result[0] = new uint[SIMD.AlignSize];
        int pos = 0;
        foreach (ref var item in result.AsSpan(1))
        {
            (item = prev.ToArray())[pos++] = uint.MaxValue;
        }
        return result;
    }
}
public static class SpanExtensions
{
    public static Span<T> Take<T>(this ref Span<T> span, int index) => Take(span, index, out span);
    public static Span<T> Skip<T>(this ref Span<T> span, int index) => Skip(span, index, out span);
    public static ReadOnlySpan<T> Take<T>(this ref ReadOnlySpan<T> span, int index) => Take(span, index, out span);
    public static ReadOnlySpan<T> Skip<T>(this ref ReadOnlySpan<T> span, int index) => Skip(span, index, out span);
    public static Span<T> Take<T>(this Span<T> span, int index, out Span<T> rest)
    {
        var result = span[..index];
        rest = span[index..];
        return result;
    }
    public static Span<T> Skip<T>(this Span<T> span, int index, out Span<T> rest)
    {
        var result = span[index..];
        rest = span[..index];
        return result;
    }
    public static ReadOnlySpan<T> Take<T>(this ReadOnlySpan<T> span, int index, out ReadOnlySpan<T> rest)
    {
        var result = span[..index];
        rest = span[index..];
        return result;
    }
    public static ReadOnlySpan<T> Skip<T>(this ReadOnlySpan<T> span, int index, out ReadOnlySpan<T> rest)
    {
        var result = span[index..];
        rest = span[..index];
        return result;
    }
}
