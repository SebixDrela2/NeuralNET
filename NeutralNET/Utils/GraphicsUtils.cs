using NeutralNET.Utils;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Text;

namespace NeutralNET.Stuff;

public static class GraphicsUtils
{
    private const int Width = 16;
    private const int Height = 16;
    private const int DigitLimit = 10;
    private const int Size = Width * Height;

    public static PixelStruct[] GetDigitsDataSet(string fontName)
    {
        var result = new PixelStruct[DigitLimit];
        var c = '0';

        var font = new Font(fontName, 8, FontStyle.Regular);

        for (var i = 0; i < DigitLimit; ++i, ++c)
        {
            result[i] = GenerateBrightStruct(c, font);
        }
        return result;
    }

    private static PixelStruct GenerateBrightStruct(char digit, Font font)
    {
        using var bitMap = new Bitmap(Width, Height, PixelFormat.Format32bppArgb);
        using (var g = Graphics.FromImage(bitMap))
        {

            var str = digit.ToString();
            var charDim = g.MeasureString(str, font);

            var pos = new PointF(
                (Width / 2) - charDim.Width / 2,
                (Height / 2) - charDim.Height / 2
            );

            g.Clear(Color.Black);
            g.TextRenderingHint = TextRenderingHint.AntiAlias;
            g.DrawString(str, font, Brushes.White, pos);
            g.Flush();          
        }
        
        var result = new float[Size];
        var index = 0;
        var brightStruct = new PixelStruct(digit - '0', Size);

        for (int y = 0; y < bitMap.Height; y++)
        {
            for (int x = 0; x < bitMap.Width; x++, ++index)
            {
                var pixel = bitMap.GetPixel(x, y);

                // DO NOT FORGET
                var brightness = (pixel.GetBrightness() * 2) - 1;
                brightStruct.Values[index] = brightness;
            }
        }
        
        return brightStruct;
    }    
}
