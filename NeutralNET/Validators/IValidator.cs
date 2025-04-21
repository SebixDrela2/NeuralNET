using NeutralNET.Matrices;
using NeutralNET.Models;

namespace NeutralNET.Validators;

public interface IValidator
{
    Matrix Input { get; }  
    IModel Model { get; init; }

    abstract void Validate();
}
