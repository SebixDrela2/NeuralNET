namespace NeutralNET.Attributes;

[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct)]
internal class ArchitectureConfigAttribute : Attribute
{
    public int InputSize { get; set; }
    public int[] HiddenLayers { get; set; } = [];
    public int OutputSize { get; set; }
}
