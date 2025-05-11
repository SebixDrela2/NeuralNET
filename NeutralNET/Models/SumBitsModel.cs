using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Validators;
using System.Runtime.CompilerServices;

namespace NeutralNET.Models;

public class SumBitsModel : IModel, IValidator
{
    private const float BitTrue = 1f;
    private const float BitFalse = 0f;

    private const int BitInput = BitModelUtils.Bits;
    private const int BitOutput = BitInput + 1;
    private const int BitLimit = 1 << BitInput;
    private const int BitRows = 1 << (BitInput * 2);
    public NeuralMatrix TrainingInput { get; set; }
    public NeuralMatrix TrainingOutput { get; set; }

    public uint[] TrainingOutputStrideMask { get; }
    public Func<NeuralMatrix> Forward { get; set; } = null!;

    public SumBitsModel()
    {      
        var inputColumns = BitInput * 2;
        var outputColumns = BitOutput;

        TrainingInput = new NeuralMatrix(BitRows, inputColumns);
        TrainingOutput = new NeuralMatrix(BitRows, outputColumns);
        TrainingOutputStrideMask = TrainingOutput.StrideMasks;
    }

    public void Prepare()
    {
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

    public void Validate()
    {
        var trainingInput = new List<float>();
        (int Correct, int Incorrect, int Total) counters = default;
        counters.Total = BitLimit * BitLimit;

        Span<float> inputSpan = TrainingInput.SpanWithGarbage;

        for (var a = 0; a < BitLimit; a++)
        {
            for (var b = 0; b < BitLimit; b++)
            {
                var sum = a + b;
                var aBits = Convert.ToString(a, 2).PadLeft(BitInput, '0').Select(x => x == '1' ? BitTrue : BitFalse);
                var bBits = Convert.ToString(b, 2).PadLeft(BitInput, '0').Select(x => x == '1' ? BitTrue : BitFalse);
                var sumBits = Convert.ToString(sum, 2).PadLeft(BitOutput, '0').Select(x => x == '1' ? BitTrue : BitFalse).ToArray();

                trainingInput.Clear();
                trainingInput.AddRange(aBits);
                trainingInput.AddRange(bBits);

                for (int i = 0; i < trainingInput.Count; i++)
                {
                    inputSpan[i] = trainingInput[i];
                }

                var outputData = Forward().GetRowSpan(0).ToArray();

                var aBitsText = string.Join("", aBits);
                var bBitsText = string.Join("", bBits);

                var expectedBits = string.Join("", sumBits);
                var actualBits = string.Join("", outputData.Select(element => element > 0f ? '1' : '0'));

                bool isCorrect = expectedBits == actualBits;
                ref var c = ref isCorrect ? ref counters.Correct : ref counters.Incorrect;
                ++c;

                var resultMessage = isCorrect
                    ? $"\e[92m  Correct: {aBitsText} + {bBitsText} = {expectedBits}, Predicted: {actualBits}\e[0m"
                    : $"\e[91mIncorrect: {aBitsText} + {bBitsText} = {expectedBits}, Predicted: {actualBits}\e[0m";

                Console.WriteLine(resultMessage);
            }
        }

        Console.WriteLine($"Validation: {counters.Correct}/{counters.Total} correct");
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

            target[index++] = c == '1' ? BitTrue : BitFalse;
        }
    }
}