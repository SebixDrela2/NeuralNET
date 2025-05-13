using NeutralNET.Utils;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;

namespace NeutralNET.Stuff;

public static class GraphicsUtils
{
    private const int FontSize = 8;

    private const int Width = 16;
    private const int Height = 16;
    private const int UpScale = 4;

    private const int ScaleWidth = Width * UpScale;
    private const int ScaleHeight = Height * UpScale;

    private const int DigitLimit = 10;
    private const int Size = Width * Height;
    private const int RandomSeed = 0xBEEF;

    private static readonly Random _rng = new Random(RandomSeed);
    public static PixelStruct[] GetDigitsDataSet(string fontName)
    {
        Directory.CreateDirectory("Digits");

        var result = new PixelStruct[DigitLimit];
        var c = '0';

        var font = new Font(fontName, FontSize * UpScale, FontStyle.Regular);

        for (var i = 0; i < DigitLimit; ++i, ++c)
        {
            var angle = float.Lerp(-5, 5, _rng.NextSingle());
            var scaleX = float.Lerp(0.98f, 1.02f, _rng.NextSingle());
            var scaleY = float.Lerp(0.98f, 1.02f, _rng.NextSingle());

            var transformation = CreateTranformationMatrix(angle, scaleX, scaleY);

            result[i] = GenerateBrightStruct(c, font, transformation);
        }
        return result;
    }

    private static PixelStruct GenerateBrightStruct(char digit, Font font, Matrix transformation)
    {
        using var bitMap = new Bitmap(ScaleWidth, ScaleHeight, PixelFormat.Format32bppArgb);
        using var trueBitMap = new Bitmap(Width, Height, PixelFormat.Format32bppArgb);

        using (var g = Graphics.FromImage(bitMap))
        {

            var str = digit.ToString();
            var fontDim = g.MeasureString(str, font);

            var pos = new PointF(
                (ScaleWidth / 2) - fontDim.Width / 2,
                (ScaleHeight / 2) - fontDim.Height / 2
            );

            var oldTransform = g.Transform;

            g.Clear(Color.Black);
            g.TextRenderingHint = TextRenderingHint.AntiAlias;
            g.Transform = transformation;
            g.DrawString(str, font, Brushes.White, pos);
            g.Flush();          
        }

        using (var g = Graphics.FromImage(trueBitMap))
        {
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.SmoothingMode = SmoothingMode.HighQuality;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;

            g.DrawImage(
                bitMap,
                new Rectangle(0, 0, Width, Height),
                new Rectangle(0, 0, ScaleWidth, ScaleHeight),
                GraphicsUnit.Pixel
            );
        }

        var result = new float[Size];
        var index = 0;
        var brightStruct = new PixelStruct(digit - '0', Size);

        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++, ++index)
            {
                var pixel = trueBitMap.GetPixel(x, y);
                var brightness = pixel.GetBrightness();

                brightStruct.Values[index] = brightness;
            }
        }

        return brightStruct;
    }

    private static Matrix CreateTranformationMatrix(float angle, float scaleX, float scaleY)
    {
        var (cx, cy) = (ScaleWidth/2f, ScaleHeight/2f);
        var m = new Matrix();

        m.Translate(-cx, -cy);
        m.Rotate(angle);
        m.Scale(scaleX, scaleY);
        m.Translate(cx, cy);

        return m;
    }
}
