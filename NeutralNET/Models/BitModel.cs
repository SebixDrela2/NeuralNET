using NeutralNET.Matrices;

namespace NeutralNET.Models;

public class BitModel : IModel
{
    private const int BitInput = 3;
    private const int BitOutput = BitInput * 2;
    private const int BitLimit = 1 << BitInput;
    private const int BitRows = 1 << BitOutput;

    public Matrix TrainingInput { get;  set; }
    public Matrix TrainingOutput { get; set ; }

    public BitModel()
    {
        var trainingInput = new List<float>();
        var trainingOutput = new List<float>();

        for (var a = 0; a < BitLimit;  a++)
        {
            for (var b = 0; b < BitLimit; b++)
            {
                var sum = a + b;
                var aBit = $"{a:b3}".Select(x => x is '1' ? 1f : 0f);
                var bBit = $"{b:b3}".Select(x => x is '1' ? 1f : 0f);
                var sumBit = $"{sum:b6}".Select(x => x is '1' ? 1f : 0f);

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
