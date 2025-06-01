using NeutralNET.Framework.Neural;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Validators;

namespace NeutralNET.Models;

public class DigitModel : IModel, IValidator
{
    private const int VariantFontCount = 10;
    public const int PixelCount = 16 * 16;
    public const int DigitLimit = 10;

    public NeuralMatrix TrainingInput { get; set; }
    public NeuralMatrix TrainingOutput { get; set; }

    private readonly string[] _fontNames = 
        ["Arial", 
        "Arial Black",
        "Bahnschrift SemiBold",
        "Courier New",
        "Curlz MT",
        "BlizzardGlobal",
        "Georgia",
        "Helvetica",
        "Century",
        "Bahnschrift Light",
        "Cambria",
        "Carlito",
        "Rockwell",
        "Symbol",
        "Rubik",
        "Times New Roman", 
        "Trebuchet MS", 
        "Verdana",
        "Vladimir Script",        
        "Tahoma", 
        "Palatino Linotype", 
        "Lucida Console", 
        "Comic Sans MS", 
        "Impact",
        "System",
        "Lucida Sans Unicode",
        "Cascadia Code",
        "Candara",
        "Calibri"];      
    private readonly int _rowCount;

    public DigitModel()
    {
        _rowCount = _fontNames.Length * DigitLimit * VariantFontCount;

        TrainingInput = new NeuralMatrix(_rowCount, PixelCount);
        TrainingOutput = new NeuralMatrix(_rowCount, 1);    
    }

    public void Prepare()
    {
        var index = 0;

        for (var k = 0; k < VariantFontCount; k++)
        {
            for (var j = 0; j < _fontNames.Length; j++)
            {
                var pixelStructs = GraphicsUtils.GetDigitsDataSet(_fontNames[j]);

                for (var pixelIndex = 0; pixelIndex < DigitLimit; ++pixelIndex, ++index)
                {
                    var inputRow = TrainingInput.GetRowSpan(index);
                    var pixelStruct = pixelStructs[pixelIndex];

                    pixelStruct.Values.CopyTo(inputRow);

                    var outputCell = TrainingOutput.GetRowSpan(index);
                    outputCell[0] = pixelStruct.MappedValue;
                }
            }
        }
    }

    public void Validate(NeuralForward forward)
    {
        var pixelStructs = GraphicsUtils.GetDigitsDataSet("BlizzardGlobal");
        var inputRow = TrainingInput.GetRowSpan(0);

        Console.WriteLine();

        for (var i = 0; i < DigitLimit; ++i)
        {           
            var pixelStruct = pixelStructs[i];

            pixelStruct.Values.CopyTo(inputRow);

            var actual = forward().GetRowSpan(0)[0];
            var expected = pixelStruct.MappedValue;

            Console.WriteLine($"ACTUAL: {actual,9:F6}, EXPECTED: {expected,9:F6}, DIFF: {(actual - expected)*9,7:F4}");
        }
    }
}
