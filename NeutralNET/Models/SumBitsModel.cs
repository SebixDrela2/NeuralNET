using NeutralNET.Matrices;
using NeutralNET.Stuff;
using System.Runtime.CompilerServices;

namespace NeutralNET.Models;

public class SumBitsModel : IModel
{
    private const int BitInput = BitModelUtils.Bits;
    private const int BitOutput = BitInput + 1;
    private const int BitLimit = 1 << BitInput;
    private const int BitRows = 1 << (BitInput * 2);
    public Matrix TrainingInput { get; set; }
    public Matrix TrainingOutput { get; set; }

    public uint[] TrainingOutputStrideMask { get; }
    public SumBitsModel()
    {      
        var inputColumns = BitInput * 2;
        var outputColumns = BitOutput;

        //outputColumns += (outputColumns & UnalignedBits) != 0 ? 32 : 0;
        //outputColumns &= ~UnalignedBits;

        TrainingInput = new Matrix(BitRows, inputColumns);
        TrainingOutput = new Matrix(BitRows, outputColumns);
        TrainingOutputStrideMask = TrainingOutput.StrideMask;

        var inputSpan = TrainingInput;
        var outputSpan = TrainingOutput.SpanWithGarbage;

        var index = 0;

        for (var a = 0; a < BitLimit; a++)
        {
            for (var b = 0; b < BitLimit; b++, ++index)
            {
                var sum = a + b;
                var rowSpanInput = TrainingInput.GetRowSpan(index);

                ConvertToBits(a, BitInput, rowSpanInput[..BitInput]);
                ConvertToBits(b, BitInput, rowSpanInput[BitInput..]);

                ConvertToBits(sum, BitOutput, TrainingOutput.GetRowSpan(index));               
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ConvertToBits(int number, int bitLength, Span<float> target)
    {
        string binary = Convert.ToString(number, 2).PadLeft(bitLength, '0');
        var index = 0;

        foreach (char c in binary)
        {
            if (index >= target.Length)
                throw new InvalidOperationException($"Index {index} out of bounds for target span of length {target.Length}");

            target[index++] = c == '1' ? 1f : 0f;
        }
    }
}