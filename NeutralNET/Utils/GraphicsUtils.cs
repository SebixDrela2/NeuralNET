using NeutralNET.Utils;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace NeutralNET.Stuff;

public static partial class GraphicsUtils
{
    private const int FontSize = Height/2;
    private const int UpScale = 4;

    private const int ScaleWidth = Width * UpScale;
    private const int ScaleHeight = Height * UpScale;

    private const int DigitLimit = 10;
    private const int Size = Width * Height;
    private const int RandomSeed = 0xBEEF;

    private static readonly Random _rng = new(RandomSeed);

    public const int Width = 16;
    public const int Height = 16;

    public const int PixelCount = Width * Height;

    [SupportedOSPlatformGuard("windows6.1")]
    public static bool IsSupported => OperatingSystem.IsWindowsVersionAtLeast(6, 1);

    public static PixelStruct[] GetDigitsDataSet(string fontName, bool applyTransformation = true)
    {
        if (!IsSupported)
        {
            throw new NotSupportedException();
        }

        Directory.CreateDirectory("Digits");

        var result = new PixelStruct[DigitLimit];
        var c = '0';

        var font = new Font(fontName, FontSize * UpScale, FontStyle.Regular);

        for (var i = 0; i < DigitLimit; ++i, ++c)
        {
            Matrix? transformation = null;

            if (applyTransformation)
            {
                var angle = float.Lerp(-10, 10, _rng.NextSingle());
                var scaleX = float.Lerp(0.95f, 1.05f, _rng.NextSingle());
                var scaleY = float.Lerp(0.95f, 1.05f, _rng.NextSingle());

                transformation = CreateTranformationMatrix(angle, scaleX, scaleY);
            }
            else
            {
                transformation = CreateTranformationMatrix(0, 1, 1);
            }

            result[i] = GenerateCharPixelStruct(c, font, transformation);
        }
        return result;
    }

    public static PixelStruct GenerateCharPixelStruct(char @char, string fontName, Matrix? transformation = null)
    {
        if (!IsSupported)
        {
            throw new NotSupportedException();
        }

        return GenerateCharPixelStruct(@char, new Font(fontName, FontSize * UpScale, FontStyle.Regular));
    }

    public static PixelStruct GenerateCharPixelStruct(char @char, Font font, Matrix? transformation = null)
    {
        if (!IsSupported)
        {
            throw new NotSupportedException();
        }

        transformation ??= new Matrix();

        using var bitMap = new Bitmap(ScaleWidth, ScaleHeight, PixelFormat.Format32bppArgb);
        using var trueBitMap = new Bitmap(Width, Height, PixelFormat.Format32bppArgb);

        using (var g = Graphics.FromImage(bitMap))
        {

            var str = @char.ToString();
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
        var brightStruct = new PixelStruct(@char - '0', Size);

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

    public static float[] LoadPixels(string path, PixelType type = PixelType.RGB)
    {
        var image = LoadImage(path);

        if (type is PixelType.RGB)
        {
            return ImageToFloatRGB(image);
        }

        return ImageToFloatGrayScale(image);
    }

    public static Bitmap LoadImage(string path)
    {
        if (!IsSupported)
        {
            throw new NotImplementedException();
        }

        return new Bitmap(Image.FromFile(path), Width, Height);
    }

    public static float[] ImageToFloatRGB(Bitmap bmp, bool normalize = true)
    {
        return ProcessImagePixels(bmp, (r, g, b) =>
        {
            if (normalize)
            {
                return [r / 255f, g / 255f, b / 255f];
            }
            return [r, g, b];
        }, channels: 3);
    }

    public static float[] ImageToFloatGrayScale(Bitmap bmp)
    {
        return ProcessImagePixels(bmp, (r, g, b) =>
        {
            return [(0.3f * r + 0.59f * g + 0.11f * b) / 255f];
        }, channels: 1);
    }

    private static float[] ProcessImagePixels(Bitmap bmp, Func<byte, byte, byte, float[]> pixelConverter, int channels)
    {
        if (!IsSupported)
        {
            throw new NotImplementedException();
        }

        var pixels = new float[Width * Height * channels];

        BitmapData data = bmp.LockBits(
            new Rectangle(0, 0, Width, Height),
            ImageLockMode.ReadOnly,
            PixelFormat.Format32bppArgb);

        try
        {
            byte[] buffer = new byte[data.Stride * Height];
            Marshal.Copy(data.Scan0, buffer, 0, buffer.Length);

            for (int y = 0; y < Height; y++)
            {
                int rowOffset = y * data.Stride;
                for (int x = 0; x < Width; x++)
                {
                    int pixelOffset = rowOffset + (x * 4);
                    int outputOffset = (y * Width + x) * channels;

                    byte b = buffer[pixelOffset];
                    byte g = buffer[pixelOffset + 1];
                    byte r = buffer[pixelOffset + 2];

                    float[] convertedValues = pixelConverter(r, g, b);
                    Array.Copy(convertedValues, 0, pixels, outputOffset, convertedValues.Length);
                }
            }
        }
        finally
        {
            bmp.UnlockBits(data);
        }

        return pixels;
    }

    private static Matrix CreateTranformationMatrix(float angle, float scaleX, float scaleY)
    {
        if (!IsSupported)
        {
            throw new NotSupportedException();
        }

        var (cx, cy) = (ScaleWidth/2f, ScaleHeight/2f);
        var m = new Matrix();

        m.Translate(-cx, -cy);
        m.Rotate(angle);
        m.Scale(scaleX, scaleY);
        m.Translate(cx, cy);

        return m;
    }
}
