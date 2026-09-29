using System.Runtime.CompilerServices;
using NeutralNET.Utils;

namespace NeutralNET.Stuff;

/// <summary>
/// Per-channel input standardization: (x - mean) / std applied to RGB
/// pixels in [0, 1] range. Statistics are cached as inverse-std so the
/// hot loop multiplies instead of divides.
///
/// Defaults assume uniformly random RGB values: mean ≈ 0.5, std ≈ 1/√12.
/// Override via <see cref="SetStats"/> or <see cref="ComputeStats"/>.
/// </summary>
public static class InputNormalization
{
    public static bool Enabled { get; set; } = true;

    public static float MeanR { get; private set; } = 0.5f;
    public static float MeanG { get; private set; } = 0.5f;
    public static float MeanB { get; private set; } = 0.5f;

    public static float InvStdR { get; private set; } = 1f / 0.2887f;
    public static float InvStdG { get; private set; } = 1f / 0.2887f;
    public static float InvStdB { get; private set; } = 1f / 0.2887f;

    private const float Eps = 1e-8f;

    public static void SetStats(
        float meanR, float stdR,
        float meanG, float stdG,
        float meanB, float stdB)
    {
        MeanR = meanR;
        MeanG = meanG;
        MeanB = meanB;

        InvStdR = 1f / (stdR < Eps ? Eps : stdR);
        InvStdG = 1f / (stdG < Eps ? Eps : stdG);
        InvStdB = 1f / (stdB < Eps ? Eps : stdB);
    }

    public static void ComputeStats(ReadOnlySpan<PixelStructRGB> samples)
    {
        if (samples.Length == 0) return;

        double sumR = 0, sumR2 = 0;
        double sumG = 0, sumG2 = 0;
        double sumB = 0, sumB2 = 0;
        long count = 0;

        foreach (ref readonly var sample in samples)
        {
            var values = sample.Pixels;
            foreach (ref readonly var px in values)
            {
                sumR += px.R; sumR2 += px.R * px.R;
                sumG += px.G; sumG2 += px.G * px.G;
                sumB += px.B; sumB2 += px.B * px.B;
                count++;
            }
        }

        if (count == 0) return;

        double meanR = sumR / count;
        double meanG = sumG / count;
        double meanB = sumB / count;

        double varR = Math.Max(0, sumR2 / count - meanR * meanR);
        double varG = Math.Max(0, sumG2 / count - meanG * meanG);
        double varB = Math.Max(0, sumB2 / count - meanB * meanB);

        SetStats(
            (float)meanR, (float)Math.Sqrt(varR),
            (float)meanG, (float)Math.Sqrt(varG),
            (float)meanB, (float)Math.Sqrt(varB));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float R, float G, float B) Denormalize(float r, float g, float b)
    => (
        r / InvStdR + MeanR,
        g / InvStdG + MeanG,
        b / InvStdB + MeanB
    );

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float R, float G, float B) Normalize(float r, float g, float b)
        => (
            (r - MeanR) * InvStdR,
            (g - MeanG) * InvStdG,
            (b - MeanB) * InvStdB
        );

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float R, float G, float B) NormalizeBgra(byte b, byte g, byte r)
    {
        const float mlt = 1.0f / 255f;
        return (
            (r * mlt - MeanR) * InvStdR,
            (g * mlt - MeanG) * InvStdG,
            (b * mlt - MeanB) * InvStdB
        );
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float R, float G, float B) ScaleOnlyBgra(byte b, byte g, byte r)
    {
        const float mlt = 1.0f / 255f;
        return (r * mlt, g * mlt, b * mlt);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (float R, float G, float B) ApplyBgra(byte b, byte g, byte r)
        => Enabled ? NormalizeBgra(b, g, r) : ScaleOnlyBgra(b, g, r);
}
