using System;
using System.IO;
using System.Linq;

namespace NeutralNET.Framework.Neural.GPT;

/// <summary>
/// Orchestrates the learning mode: prepares batches, resumes from a checkpoint
/// if one exists, runs training, and generates a sample at the end.
/// </summary>
public class GptLearner
{
    private readonly GptNeuralFramework _gpt;
    private readonly GPTTextDataLoader _loader;
    private readonly TrainingConfig _cfg;

    public GptLearner(GptNeuralFramework gpt, GPTTextDataLoader loader, TrainingConfig cfg)
    {
        _gpt = gpt;
        _loader = loader;
        _cfg = cfg;
    }

    public void Run()
    {
        var batches = _loader.GetBatches(_cfg.ContextSize)
                             .Chunk(_cfg.MaxBatchSize)
                             .ToList();

        Console.WriteLine($"[DataLoader] Created {batches.Count:N0} mini-batches (Batch Size: {_cfg.MaxBatchSize}).");

        var checkpoints = new GptCheckpointManager(_cfg.CanonicalCheckpoint, _cfg.BestLossSidecar);
        checkpoints.TryResume(_gpt, out string msg);
        Console.WriteLine(msg);

        Console.WriteLine("\n--- Starting Training ---");
        new GptTrainer(_gpt, checkpoints, _cfg, batches).Train();

        Console.WriteLine("\n--- Generating Text ---");
        string prompt = "To be or not to be";
        int[] outTokens = _gpt.Generate(_loader.Encode(prompt).ToArray(), 150, 0.8f, 0.9f);
        Console.WriteLine($"Prompt: {prompt}");
        Console.WriteLine($"Generated Output:\n{_loader.Decode(outTokens)}");
    }
}
