using NeutralNET.Framework;
using NeutralNET.Matrices;

namespace NeutralNET.Validators;

public abstract class Validator : IValidator
{
    public Matrix Input { get; init; }

    public Func<Matrix> Forward { get; init; }

    public Validator(IModelRunner modelRunner)
    {
        Input = modelRunner.Input;
        Forward = modelRunner.Forward;
    }

    public abstract void Validate();
}
