

using NeutralNET.Framework;
using NeutralNET.Models;

namespace NeutralNET.Validators;

public class ThreeBitModelValidator : Validator
{
    private const int BitInput = 3;
    private const int BitLimit = 1 << BitInput;

    public ThreeBitModelValidator(IModelRunner modelRunner) : base(modelRunner)
    {
        Model = new BitModel();
    }
    public override void Validate()
    {
        var trainingInput = new List<float>();

        for (var a = 0; a < BitLimit; a++)
        {
            for (var b = 0; b < BitLimit; b++)
            {
                var sum = a + b;
                var aBits = $"{a:b3}".Select(x => x is '1' ? 1f : 0f);
                var bBits = $"{b:b3}".Select(x => x is '1' ? 1f : 0f);

                trainingInput.Clear();
                trainingInput.AddRange(aBits);
                trainingInput.AddRange(bBits);

                Input.Data = new ArraySegment<float>([.. trainingInput]);
                var outputData = Forward().Data;

                var aBitsText = string.Join("", aBits);
                var bBitsText = string.Join("", bBits);

                var expectedBits = string.Join("", $"{b:b6}".Select(x => x is '1' ? 1f : 0f));
                var actualBits = string.Join("", outputData.Select(element => element > 0.8f ? '1' : '0'));

                var resultText = "TRUE";

                if (expectedBits != actualBits)
                {
                    resultText = "FALSE";
                }
                Console.WriteLine($"{resultText}: {aBitsText} + {bBitsText} = {expectedBits} and {actualBits}");               
            }
        }
    }
   
}
