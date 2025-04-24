using NeutralNET.Framework;
using NeutralNET.Stuff;

namespace NeutralNET.Validators;

public class SumBitsModelValidator(IModelRunner modelRunner) : Validator(modelRunner)
{
    private const int BitInput = BitModelUtils.Bits;
    private const int BitLimit = 1 << BitInput;
    
    public override void Validate()
    {
        var trainingInput = new List<float>();
        (int Correct, int Incorrect, int Total) counters = default;
        counters.Total = BitLimit * BitLimit;

        for (var a = 0; a < BitLimit; a++)
        {
            for (var b = 0; b < BitLimit; b++)
            {
                var sum = a + b;
                var aBits = Convert.ToString(a, 2).PadLeft(BitInput, '0').Select(x => x == '1' ? 1f : 0f);
                var bBits = Convert.ToString(b, 2).PadLeft(BitInput, '0').Select(x => x == '1' ? 1f : 0f);
                var sumBits = Convert.ToString(sum, 2).PadLeft(BitInput * 2, '0').Select(x => x == '1' ? 1f : 0f).ToArray();

                trainingInput.Clear();
                trainingInput.AddRange(aBits);
                trainingInput.AddRange(bBits);

                Input.Data = [.. trainingInput];
                var outputData = Forward().Data;

                var aBitsText = string.Join("", aBits);
                var bBitsText = string.Join("", bBits);

                var expectedBits = string.Join("", sumBits);
                var actualBits = string.Join("", outputData.Select(element => element > 0.8f ? '1' : '0'));

                bool isCorrect = expectedBits == actualBits;
                ref var c = ref isCorrect ? ref counters.Correct : ref counters.Incorrect; 
                ++c;
                var resultMessage = isCorrect
                    ? $"\e[92m  Correct: {aBitsText} + {bBitsText} = {expectedBits}, Predicted: {actualBits}\e[0m"
                    : $"\e[91mIncorrect: {aBitsText} + {bBitsText} = {expectedBits}, Predicted: {actualBits}\e[0m";

                // Console.WriteLine(resultMessage);
            }
        }

        var ratio = (double)counters.Correct / counters.Total;
        Console.WriteLine($"results: {counters.Correct}/{counters.Total} ({ratio:P2})");
        
        foreach (var (lines, trace) in Matrices.Matrix.Traces)
        {
            Console.WriteLine();
            Console.WriteLine(lines);
        }
        foreach (var (w, h) in Matrices.Matrix.Sizes.Order())
        {
            Console.WriteLine($"Matrix {w}x{h}");
        }
    }
   
}
