namespace NeutralNET.Framework.Neural.GPT;

public class TrainingConfig
{
    // Model
    public int ContextSize { get; set; } = 256;
    public int EmbedDim { get; set; } = 128;
    public int IntermediateDim { get; set; } = 512;
    public int NumHeads { get; set; } = 4;
    public int NumLayers { get; set; } = 2;
    public int MaxBatchSize { get; set; } = 64;

    // Training
    public float LearningRate { get; set; } = 0.0003f;
    public int TotalEpochs { get; set; } = 10;
    public int WarmupSteps { get; set; } = 20;
    public float FinalLrMultiplier { get; set; } = 0f;   // cosine floor

    // Logging
    public bool PrintBatchDiag { get; set; } = true;
    public int BatchDiagEvery { get; set; } = 1;

    // Paths
    public string BuildFolder { get; set; } = "";
    public string CorpusFile { get; set; } = "";
    public string CanonicalCheckpoint { get; set; } = "";
    public string BestLossSidecar { get; set; } = "";

    public GptConfig ToGptConfig(int vocabSize) => new()
    {
        VocabSize = vocabSize,
        ContextSize = ContextSize,
        EmbedDim = EmbedDim,
        IntermediateDim = IntermediateDim,
        NumHeads = NumHeads,
        NumLayers = NumLayers,
        LearningRate = LearningRate,
        MaxBatchSize = MaxBatchSize
    };

    public static TrainingConfig Default()
    {
        string buildFolder = Path.Combine(GlobalScope.BuildDirectory, "magnificency");

        return Default(buildFolder);
    }

    public static TrainingConfig Default(string buildFolder)
    {
        Directory.CreateDirectory(buildFolder);
        return new TrainingConfig
        {
            BuildFolder = buildFolder,
            CorpusFile = Path.Combine(buildFolder, "ShakespeareWork.txt"),
            CanonicalCheckpoint = Path.Combine(buildFolder, "shakespeare_gpt.bin"),
            BestLossSidecar = Path.Combine(buildFolder, "shakespeare_gpt.bestloss.txt")
        };
    }
}
