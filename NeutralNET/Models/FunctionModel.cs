using NeutralNET.Framework;
using NeutralNET.Framework.Neural;
using NeutralNET.Validators;

namespace NeutralNET.Models;

public class FunctionModel : IDynamicModel, IDynamicValidator
{
    public const int MaxRange = 100;
    public const int ScaleFactor = 100;
    public Func<float, float, float> PrepareFunction => (x, y) => x * x;

    public void Validate(NeuralForward forward, Architecture architecture)
    {
        var input = architecture.MatrixNeurons[0].GetRowSpan(0);
        ref var actual = ref architecture.MatrixNeurons[^1].GetRowSpan(0)[0];

        for (int i = 0; i < 100; i++)
        {
            var a = Random.Shared.NextSingle() * MaxRange;
            var b = Random.Shared.NextSingle() * MaxRange;

            input[0] = TranslateInto(a);
            input[1] = TranslateInto(b);

            forward();

            var realActual = TranslateFrom(actual);

            var expected = PrepareFunction(a, b);
            var diff = MathF.Abs(expected - realActual);

            var resultMessage = diff < 0.05
                    ? $"\e[92m  Correct:  f({a,10:F2}, {b,10:F2}) = A:{realActual,10:F2} | E:{expected,10:F2}, Diff = {diff,10:F2}\e[0m"
                    : $"\e[91mIncorrect:  f({a,10:F2}, {b,10:F2}) = A:{realActual,10:F2} | E:{expected,10:F2}, Diff = {diff,10:F2}\e[0m";

            Console.WriteLine(resultMessage);
        }
    }

    public float TranslateInto(float value)
    {
        return value / ScaleFactor;
    }

    public float TranslateFrom(float value)
    {
        return value * ScaleFactor;
    }
}
