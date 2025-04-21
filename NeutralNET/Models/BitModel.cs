using NeutralNET.Matrices;
using NeutralNET.Stuff;

namespace NeutralNET.Models;

public class BitModel : IModel
{
    private const int BitInput = BitModelUtils.BitInput;
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

                var aBits = Convert.ToString(a, 2).PadLeft(BitInput, '0').Select(x => x == '1' ? 1f : 0f);
                var bBits = Convert.ToString(b, 2).PadLeft(BitInput, '0').Select(x => x == '1' ? 1f : 0f);
                var sumBits = Convert.ToString(sum, 2).PadLeft(BitInput * 2, '0').Select(x => x == '1' ? 1f : 0f).ToArray();

                trainingInput.AddRange(aBits);
                trainingInput.AddRange(bBits);
                trainingOutput.AddRange(sumBits);

                Console.WriteLine($"Training input: {string.Join("", aBits)} + {string.Join("", bBits)} = {string.Join("", sumBits)}");
            }
        }

        TrainingInput = new Matrix(BitRows, BitOutput)
        {
            Data = trainingInput.ToArray()
        };

        TrainingOutput = new Matrix(BitRows, BitOutput)
        {
            Data = trainingOutput.ToArray()
        };
    }
}
