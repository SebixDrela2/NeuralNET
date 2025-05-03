using NeutralNET.Framework;
using NeutralNET.Stuff;

namespace NeutralNET.Validators;

public class SumBitsModelValidator(IModelRunner modelRunner) : Validator(modelRunner)
{
    private const int BitInput = BitModelUtils.Bits;
    private const int BitLimit = 1 << BitInput;
    private const int BitOutput = BitInput + 1;

    public override void Validate()
    {
        var trainingInput = new List<float>();
        (int Correct, int Incorrect, int Total) counters = default;
        counters.Total = BitLimit * BitLimit;
        
        Span<float> inputSpan = Input.SpanWithGarbage;

        for (var a = 0; a < BitLimit; a++)
        {
            for (var b = 0; b < BitLimit; b++)
            {
                var sum = a + b;
                var aBits = Convert.ToString(a, 2).PadLeft(BitInput, '0').Select(x => x == '1' ? 1f : 0f);
                var bBits = Convert.ToString(b, 2).PadLeft(BitInput, '0').Select(x => x == '1' ? 1f : 0f);
                var sumBits = Convert.ToString(sum, 2).PadLeft(BitOutput, '0').Select(x => x == '1' ? 1f : 0f).ToArray();
              
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
                var actualBits = string.Join("", outputData.Select(element => element > 0.8f ? '1' : '0'));

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

}
