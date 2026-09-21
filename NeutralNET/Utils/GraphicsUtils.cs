using NeutralNET.Utils;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace NeutralNET.Stuff;

public static partial class GraphicsUtils
{
    public const int FontSize = Height / 2;
    public const int UpScale = 4;

    private const int ScaleWidth = Width * UpScale;
    private const int ScaleHeight = Height * UpScale;

    private const int DigitLimit = 10;
    private const int Size = Width * Height;

    public const int Width = 64;
    public const int Height = 64;

    public const int PixelCount = Width * Height;

    private static SizeF ScaleSize { [MethodImpl(Inline)] get => new(ScaleWidth, ScaleHeight); }

    [SupportedOSPlatformGuard("windows6.1")]
    public static bool IsSupported => OperatingSystem.IsWindowsVersionAtLeast(6, 1);

    #region Letter Data Generation (NEW)

    public static readonly char[] DefaultLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".ToCharArray();

    public static PixelStructRGB[] GetLettersDataSetRGB(string fontName, bool applyTransformation, FontStyle style = default)
        => GetLettersDataSetRGB(fontName, DefaultLetters, applyTransformation, style);

    public static PixelStructRGB[] GetLettersDataSetRGB(string fontName, char[] characters, bool applyTransformation, FontStyle style = default)
    {
        if (!IsSupported) throw new NotSupportedException();

        var result = new PixelStructRGB[characters.Length];
        using var font = new Font(fontName, FontSize * UpScale, style);

        Parallel.For(0, characters.Length, i =>
        {
            var transformation = applyTransformation
                ? new(
                    Angle: float.Lerp(-5, 5, Random.Shared.NextSingle()),
                    Scale: (
                        X: float.Lerp(0.95f, 1.05f, Random.Shared.NextSingle()),
                        Y: float.Lerp(0.95f, 1.05f, Random.Shared.NextSingle())
                    )
                )
                : ImageTransformation.None;

            result[i] = GenerateCharPixelStructRGB(characters[i], font, i, transformation);
        });

        return result;
    }

    public static PixelStructRGB GenerateCharPixelStructRGB(char @char, Font font, int classLabel, ImageTransformation transformation)
    {
        if (!IsSupported) throw new NotSupportedException();

        using var bitMap = new Bitmap(ScaleWidth, ScaleHeight, PixelFormat.Format32bppArgb);
        using var trueBitMap = new Bitmap(Width, Height, PixelFormat.Format32bppArgb);

        using (var g = Graphics.FromImage(bitMap))
        {
            var str = @char.ToString();
            var fontDim = g.MeasureString(str, font);

            var pos = new PointF(
                (ScaleWidth / 2f) - fontDim.Width / 2f,
                (ScaleHeight / 2f) - fontDim.Height / 2f
            );

            var (background, letter) = Color.GetRandomColors();

            g.Clear(background);
            g.TextRenderingHint = TextRenderingHint.AntiAlias;
            g.Transform = transformation.ToMatrix();
            g.DrawString(str, font, new SolidBrush(letter), pos);
            g.Flush();
        }

        using (var g = Graphics.FromImage(trueBitMap))
        {
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.SmoothingMode = SmoothingMode.HighQuality;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            g.CompositingQuality = CompositingQuality.HighQuality;

            g.DrawImage(
                bitMap,
                new Rectangle(0, 0, Width, Height),
                new Rectangle(0, 0, ScaleWidth, ScaleHeight),
                GraphicsUnit.Pixel
            );
        }

        var pixels = new PixelStructRGB(classLabel, Size);

        // Lock bitmap bits for fast memory extraction instead of calling GetPixel
        var data = trueBitMap.LockBits(
            new Rectangle(0, 0, Width, Height),
            ImageLockMode.ReadOnly,
            PixelFormat.Format32bppArgb
        );

        try
        {
            byte[] buffer = new byte[data.Stride * Height];
            Marshal.Copy(data.Scan0, buffer, 0, buffer.Length);

            int index = 0;
            for (int y = 0; y < Height; y++)
            {
                int rowOffset = y * data.Stride;
                for (int x = 0; x < Width; x++, index++)
                {
                    int pixelOffset = rowOffset + (x * 4);
                    byte b = buffer[pixelOffset];
                    byte g = buffer[pixelOffset + 1];
                    byte r = buffer[pixelOffset + 2];

                    pixels.Values[index] = (r / 255.0f, g / 255.0f, b / 255.0f);
                }
            }
        }
        finally
        {
            trueBitMap.UnlockBits(data);
        }

        return pixels;
    }

    #endregion

    #region Original Digit Data Generation (PRESERVED)

    public static PixelStructRGB[] GetDigitsDataSetRGB(string fontName) => GetDigitsDataSetRGB(fontName, true, default);
    public static PixelStructRGB[] GetDigitsDataSetRGB(string fontName, bool applyTransformation, FontStyle style = default)
    {
        if (!IsSupported)
        {
            throw new NotSupportedException();
        }

        var result = new PixelStructRGB[DigitLimit];
        var c = '0';

        using var font = new Font(fontName, FontSize * UpScale, style);

        for (var i = 0; i < DigitLimit; ++i, ++c)
        {
            var transformation = applyTransformation
                ? new(float.Lerp(-5, 5, Random.Shared.NextSingle()))
                : ImageTransformation.None;

            result[i] = GenerateCharPixelStructRGB(c, font, transformation);
        }
        return result;
    }

    public static PixelStruct[] GetDigitsDataSet(string fontName) => GetDigitsDataSet(fontName, true, default);
    public static PixelStruct[] GetDigitsDataSet(string fontName, bool applyTransformation, FontStyle style = default)
    {
        if (!IsSupported) throw new NotSupportedException();

        var result = new PixelStruct[DigitLimit];
        var c = '0';

        using var font = new Font(fontName, FontSize * UpScale, style);

        for (var i = 0; i < DigitLimit; ++i, ++c)
        {
            var transformation = applyTransformation
                ? new(float.Lerp(-5, 5, Random.Shared.NextSingle()))
                : ImageTransformation.None;

            result[i] = GenerateCharPixelStruct(c, font, transformation);
        }
        return result;
    }

    public static PixelStructRGB GenerateCharPixelStructRGB(char ch, Font font, ImageTransformation transformation)
    {
        if (!IsSupported) throw new NotSupportedException();

        using var bitMap = new Bitmap(ScaleWidth, ScaleHeight, PixelFormat.Format32bppArgb);
        using var trueBitMap = new Bitmap(Width, Height, PixelFormat.Format32bppArgb);

        using (var g = Graphics.FromImage(bitMap))
        {
            var pos = ((g.MeasureString([ch], font) - ScaleSize) * 0.5f).ToPointF();

            g.Clear(Color.Black);
            g.TextRenderingHint = TextRenderingHint.AntiAlias;
            g.Transform = transformation.ToMatrix();
            g.DrawString([ch], font, Brushes.White, pos);
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

        var index = 0;
        var pixels = new PixelStructRGB(ch - '0', Size);

        for (int y = 0; y < Height; y++)
        {
            for (int x = 0; x < Width; x++, ++index)
            {
                var pixel = trueBitMap.GetPixel(x, y);
                pixels.Values[index] = (pixel.R, pixel.G, pixel.B);
            }
        }

        return pixels;
    }

    public static PixelStruct GenerateCharPixelStruct(char @char, string fontName, FontStyle style = default)
    {
        if (!IsSupported) throw new NotSupportedException();

        using var font = new Font(fontName, FontSize * UpScale, style);
        return GenerateCharPixelStruct(@char, font);
    }
    public static PixelStruct GenerateCharPixelStruct(char @char, Font font) => GenerateCharPixelStruct(@char, font, ImageTransformation.None);
    public static PixelStruct GenerateCharPixelStruct(char ch, Font font, ImageTransformation transformation)
    {
        if (!IsSupported) throw new NotSupportedException();

        using var bitMap = new Bitmap(ScaleWidth, ScaleHeight, PixelFormat.Format32bppArgb);
        using var trueBitMap = new Bitmap(Width, Height, PixelFormat.Format32bppArgb);

        using (var g = Graphics.FromImage(bitMap))
        {
            var pos = ((g.MeasureString([ch], font) - ScaleSize) * 0.5f).ToPointF();

            g.Clear(Color.Black);
            g.TextRenderingHint = TextRenderingHint.AntiAlias;
            g.Transform = transformation.ToMatrix();
            g.DrawString([ch], font, Brushes.White, pos);
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

        var index = 0;
        var brightStruct = new PixelStruct(ch - '0', Size);

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

    #region SLOW AF / DEPRECATED / WEIRD

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
        if (!IsSupported) throw new NotImplementedException();
        return new Bitmap(Image.FromFile(path), Width, Height);  // Dispose Image?
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
        if (!IsSupported) throw new NotImplementedException();

        var pixels = new float[Width * Height * channels];

        BitmapData data = bmp.LockBits(
            new Rectangle(0, 0, Width, Height),
            ImageLockMode.ReadOnly,
            PixelFormat.Format32bppArgb
        );

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
    #endregion

    public record struct ImageTransformation(float Angle, (float X, float Y) Scale)
    {
        public static ImageTransformation None => new(0, (1, 1));

        public ImageTransformation(float angle) : this(angle, (1, 1)) { }
        public readonly Matrix ToMatrix()
        {
            if (!IsSupported) throw new NotSupportedException();

            var (cx, cy) = (ScaleWidth * 0.5f, ScaleHeight * 0.5f);
            var m = new Matrix();

            try
            {
                m.Translate(-cx, -cy);
                m.Rotate(Angle);
                m.Scale(Scale.X, Scale.Y);
                m.Translate(cx, cy);

                return m;
            }
            catch
            {
                m.Dispose();
                throw;
            }
        }
    }

    #endregion

    public static PixelStructRGB GetPixels(Bitmap bmp)
    {
        if (!IsSupported) throw new NotSupportedException();
        var pixels = new PixelStructRGB(default, bmp.Size.Height * bmp.Size.Width);
        byte[] buff;

        var data = bmp.LockBits(
            new Rectangle(Point.Empty, bmp.Size),
            ImageLockMode.ReadOnly,
            PixelFormat.Format32bppArgb
        );
        try
        {
            buff = new byte[data.Stride * data.Height];
            Marshal.Copy(data.Scan0, buff, 0, buff.Length);
        }
        finally
        {
            bmp.UnlockBits(data);
        }

        const float mlt = 1.0f / 255;

        var src = buff.AsSpan();
        var dst = pixels.Pixels;

        for (int y = 0, i = 0, pos = 0; y < data.Height; ++y, pos += data.Stride)
        {
            for (int x = 0, posX = pos; x < data.Width; ++x, ++i, posX += 4)
            {
                dst[i] = (
                    float.Clamp(src[posX + 2] * mlt, 0, 1),
                    float.Clamp(src[posX + 1] * mlt, 0, 1),
                    float.Clamp(src[posX + 0] * mlt, 0, 1)
                );
            }
        }

        return pixels;
    }
}
