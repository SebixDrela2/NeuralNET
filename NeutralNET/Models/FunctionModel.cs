using NeutralNET.Framework;
using NeutralNET.Framework.Neural;
using NeutralNET.Validators;

namespace NeutralNET.Models;

public class FunctionModel : IDynamicModel, IDynamicValidator
{
    public Func<float, float, float> PrepareFunction => (x, _) => x * x;

    public void Validate(NeuralForward forward, Architecture architecture)
    {
        var input = architecture.MatrixNeurons[0].GetRowSpan(0);
        ref var actual = ref architecture.MatrixNeurons[^1].GetRowSpan(0)[0];

        for (int i = 0; i < 100; i++)
        {
            var notScaledInput = Random.Shared.NextSingle() * 10;
            var notScaledInput2 = Random.Shared.NextSingle() * 10;

            input[0] = ScaleDown(notScaledInput);
            input[1] = ScaleDown(notScaledInput2);

            forward();

            var scaledActual = ScaleUp(ScaleUp(actual));
            var expected = PrepareFunction(notScaledInput, notScaledInput2);

            var diff = MathF.Abs(expected - scaledActual);

            var resultMessage = diff < 0.05
                    ? $"\e[92m  Correct:  f({notScaledInput,10:F2}, {notScaledInput2,10:F2}) = A:{scaledActual,10:F2} | E:{expected,10:F2}, Diff = {diff,10:F2}\e[0m"
                    : $"\e[91mIncorrect:  f({notScaledInput,10:F2}, {notScaledInput2,10:F2}) = A:{scaledActual,10:F2} | E:{expected,10:F2}, Diff = {diff,10:F2}\e[0m";

            Console.WriteLine(resultMessage);
        }
    }

    public float ScaleDown(float value)
    {
        return value / 1;
    }

    public float ScaleUp(float value)
    {
        return value * 1;
    }
}
