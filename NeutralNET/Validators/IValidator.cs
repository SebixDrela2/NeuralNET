using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Connected.Neural;

namespace NeutralNET.Validators;

public interface IValidator
{
    void Validate(NeuralForward forward);
}

public interface IDynamicValidator
{
    void Validate(NeuralForward forward, Architecture architecture);
}
