using NeutralNET.Matrices;

namespace NeutralNET.Models;

public class BitModel : IModel
{

    private const int BitInput = 4;
    private const int BitOutput = BitInput * 2;
    private const int BitLimit = 1 << BitInput;
    private const int BitRows = 1 << BitOutput;
    public Matrix TrainingInput { get;  init; }
    public Matrix TrainingOutput { get; init ; }

    public BitModel()
    {
        var trainingInput = new List<float>();
        var trainingOutput = new List<float>();

        var index = 0;

        for (var a = 0; a < BitLimit;  a++)
        {
            for (var b = 0; b < BitLimit; b++, index++)
            {
                var sum = a + b;
                var aBit = $"{a:b4}".Select(x => x is '1' ? 1f : 0f);
                var bBit = $"{b:b4}".Select(x => x is '1' ? 1f : 0f);
                var sumBit = $"{b:b8}".Select(x => x is '1' ? 1f : 0f);

                trainingInput.AddRange(aBit);
                trainingInput.AddRange(bBit);

                trainingOutput.AddRange(sumBit);
            }
        }

        TrainingInput = new Matrix(BitRows, BitOutput)
        {
            Data = new ArraySegment<float>(trainingInput.ToArray())
        };

        TrainingOutput = new Matrix(BitRows, BitOutput)
        {
            Data = new ArraySegment<float>(trainingOutput.ToArray())
        };
    }
}
