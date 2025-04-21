using NeutralNET.Framework;
using NeutralNET.Stuff;

namespace NeutralNET.Validators;

public class BitModelValidator(IModelRunner modelRunner) : Validator(modelRunner)
{
    private const int BitInput = BitModelUtils.BitInput;
    private const int BitLimit = 1 << BitInput;
    
    public override void Validate()
    {
        var trainingInput = new List<float>();

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

                Input.Data = new ArraySegment<float>([.. trainingInput]);
                var outputData = Forward().Data;

                var aBitsText = string.Join("", aBits);
                var bBitsText = string.Join("", bBits);

                var expectedBits = string.Join("", sumBits);
                var actualBits = string.Join("", outputData.Select(element => element > 0.8f ? '1' : '0'));
                
                var resultText = "TRUE";

                if (expectedBits != actualBits)
                {
                    resultText = "FALSE";
                }

                var resultMessage = resultText == "TRUE"
                    ? $"Correct: {aBitsText} + {bBitsText} = {expectedBits}, Predicted: {actualBits}"
                    : $"Incorrect: {aBitsText} + {bBitsText} = {expectedBits}, Predicted: {actualBits}";

                Console.WriteLine(resultMessage);
            }
        }
    }
   
}
