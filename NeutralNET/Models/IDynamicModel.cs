namespace NeutralNET.Models;

public interface IDynamicModel
{
    Func<float, float, float> PrepareFunction { get; }
    float ScaleDown(float value);
    float ScaleUp(float value);
}
