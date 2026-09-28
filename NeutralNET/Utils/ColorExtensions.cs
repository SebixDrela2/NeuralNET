using System.Drawing;

namespace NeutralNET.Utils;

public static class ColorExtensions
{
    extension(Color color)
    {
        public static (Color Background, Color Letter) GetRandomColors()
        {
            var r = GetRandomByteColor();
            var g = GetRandomByteColor();
            var b = GetRandomByteColor();

            var background = Color.FromArgb(r.Background, g.Background, b.Background);
            var letter = Color.FromArgb(r.Letter, g.Letter, b.Letter);

            return (background, letter);
        }
    }

    private static (byte Background, byte Letter) GetRandomByteColor()
    {
        var (b, l) = GetRandomFloatColor();

        return (byte.CreateSaturating(b * 0xFF), byte.CreateSaturating(l * 0xFF));
    }

    private static (float Background, float Letter) GetRandomFloatColor()
    {
        var distance = Random.Shared.NextSingle() * 0.7f + 0.3f;
        var offset = Random.Shared.NextSingle() * (1 - distance);
        var isLight = Random.Shared.Next(0, 2) == 0;

        if (isLight)
        {
            return (offset, offset + distance);
        }

        return (offset + distance, offset);
    }
}
