using NeutralNET.Matrices;

namespace NeutralNET.Validators;

public interface IValidator
{
    Matrix Input { get; }  

    abstract void Validate();
}
