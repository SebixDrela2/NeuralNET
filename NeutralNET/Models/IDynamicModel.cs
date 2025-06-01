namespace NeutralNET.Models;

public interface IDynamicModel
{
    Func<float, float, float> PrepareFunction { get; }
    float TranslateInto(float value);
    float TranslateFrom(float value);
}
