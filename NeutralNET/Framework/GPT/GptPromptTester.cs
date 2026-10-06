using System;
using System.IO;

namespace NeutralNET.Framework.Neural.GPT;

/// <summary>
/// Loads a checkpoint into an existing model and lets you type prompts,
/// then generates continuations.
/// </summary>
public class GptPromptTester
{
    private readonly GptNeuralFramework _gpt;
    private readonly GPTTextDataLoader _loader;

    public GptPromptTester(GptNeuralFramework gpt, GPTTextDataLoader loader, string checkpointPath)
    {
        _gpt = gpt;
        _loader = loader;

        if (!File.Exists(checkpointPath))
            throw new FileNotFoundException($"[Tester] Checkpoint not found: {checkpointPath}");

        GptTrainingRunner.LoadModel(gpt, checkpointPath);
        Console.WriteLine($"[Tester] Loaded checkpoint: {checkpointPath}");
    }

    public void Run()
    {
        Console.WriteLine("\n=== Interactive Tester ===");
        Console.WriteLine("Commands:  :q  quit    :t <temp>  set temperature    :p <topP>  set top-p");
        Console.WriteLine();

        float temperature = 0.8f;
        float topP = 0.9f;

        while (true)
        {
            Console.Write($"prompt [t={temperature:F2} p={topP:F2}]> ");
            string? line = Console.ReadLine();
            if (line == null) break;

            if (line == ":q") break;
            if (line.StartsWith(":t "))
            {
                if (float.TryParse(line[3..], out float t)) temperature = MathF.Max(0.01f, t);
                continue;
            }
            if (line.StartsWith(":p "))
            {
                if (float.TryParse(line[3..], out float p)) topP = Math.Clamp(p, 0.01f, 1f);
                continue;
            }

            if (string.IsNullOrEmpty(line)) line = "To be or not to be";

            int[] promptTokens = _loader.Encode(line).ToArray();
            int[] outputTokens = _gpt.Generate(promptTokens, maxNewTokens: 200, temperature, topP);
            string output = _loader.Decode(outputTokens);

            Console.WriteLine();
            Console.WriteLine(output);
            Console.WriteLine();
        }
    }
}
