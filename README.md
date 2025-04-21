## How to run/validate.

### Run
In order to properly run a neural network, you must specify it's config parameters.
Most importantly network architecture, model, batchsize, epochs (training count) and the learning rate.

**NeuralNetwork** example

```cs
using NeutralNET.Framework;
using NeutralNET.Models;
using NeutralNET.Validators;

namespace NeutralTest;

internal class Program
{
    static void Main(string[] args)
    {
        var network = new NeuralNetworkBuilder()
            .WithArchitecture(12, 32, 32, 12)
            .WithEpochs(2000)
            .WithBatchSize(200)
            .WithLearningRate(0.01f)
            .WithWeightDecay(1e-5f)
            .WithModel(new SumBitsModel())
            .Build();

        var modelRunner = network.Run();
        var validator = new SumBitsModelValidator(modelRunner);
        validator.Validate();
    }
}

```

### Validate 
To validate, run network does returns the `ModelRunner` which most importantly has the `Input` in which one can set the data.
and `Forward` which executes the model and returns the `OutPutMatrix`

```cs
public class SumBitsModelValidator(IModelRunner modelRunner) : Validator(modelRunner)
{
    private const int BitInput = BitModelUtils.Bits;
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

                Input.Data = [.. trainingInput];
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
```
