using NeutralNET.Matrices;
using NeutralNET.Stuff;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace NeutralNET.Models;

public class SumBitsModel : IModel
{
    private const int BitInput = BitModelUtils.Bits;
    private const int BitOutput = BitInput * 2;
    private const int BitLimit = 1 << BitInput;
    private const int BitRows = 1 << BitOutput;
    public Matrix TrainingInput { get; set; }
    public Matrix TrainingOutput { get; set; }

    public SumBitsModel()
    {
        var inputColumns = BitInput * 2;
        var outputColumns = BitOutput;

        //outputColumns += (outputColumns & UnalignedBits) != 0 ? 32 : 0;
        //outputColumns &= ~UnalignedBits;

        TrainingInput = new Matrix(BitRows, inputColumns);
        TrainingOutput = new Matrix(BitRows, outputColumns);

        var inputSpan = TrainingInput.Span;
        var outputSpan = TrainingOutput.Span;

        int inputIndex = 0;
        int outputIndex = 0;

        for (var a = 0; a < BitLimit; a++)
        {
            for (var b = 0; b < BitLimit; b++)
            {
                var sum = a + b;

                ConvertToBits(a, BitInput, inputSpan, ref inputIndex);
                ConvertToBits(b, BitInput, inputSpan, ref inputIndex);

                ConvertToBits(sum, BitOutput, outputSpan, ref outputIndex);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ConvertToBits(int number, int bitLength, Span<float> target, ref int index)
    {
        string binary = Convert.ToString(number, 2).PadLeft(bitLength, '0');
        foreach (char c in binary)
        {
            if (index >= target.Length)
                throw new InvalidOperationException($"Index {index} out of bounds for target span of length {target.Length}");

            target[index++] = c == '1' ? 1f : 0f;
        }
    }
}