using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;

namespace NeutralNET.Utils;

public struct PixelStructRGB(int label, int size)
{
    public int Label = label;
    public ColorRGB[] Values = new ColorRGB[size];
    public readonly Span<ColorRGB> Pixels => Values;
    public readonly Span<float> Flat => MemoryMarshal.Cast<ColorRGB, float>(Pixels);
}

[InlineArray(3)]
public struct ColorRGB
{
    private float _elem;

    public ref float R { [MethodImpl(Inline), UnscopedRef] get => ref this[0]; }
    public ref float G { [MethodImpl(Inline), UnscopedRef] get => ref this[1]; }
    public ref float B { [MethodImpl(Inline), UnscopedRef] get => ref this[2]; }
    public Span<float> Span { [MethodImpl(Inline), UnscopedRef] get => this; }

    public static implicit operator ColorRGB((float R, float G, float B) color) => new() { R = color.R, G = color.G, B = color.B };
}
