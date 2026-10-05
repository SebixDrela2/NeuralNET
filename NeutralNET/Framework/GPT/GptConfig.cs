namespace NeutralNET.Framework.Neural.GPT;

public class GptConfig
{
    public required int VocabSize { get; set; }
    public required int ContextSize { get; set; }
    public required int MaxBatchSize { get; set; }
    public required int EmbedDim { get; set; }
    public required int IntermediateDim { get; set; }
    public required int NumHeads { get; set; }
    public required int NumLayers { get; set; }
    public required float LearningRate { get; set; }
}
