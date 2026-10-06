using System.Globalization;

namespace NeutralNET.Framework.Neural.GPT;

/// <summary>
/// Wraps save/load with per-epoch snapshots, best-model tracking, and a sidecar
/// that persists bestLoss + globalStep across process restarts.
/// </summary>
public class GptCheckpointManager
{
    private readonly string _canonicalPath;
    private readonly string _sidecarPath;

    public float BestLoss { get; private set; } = float.PositiveInfinity;
    public int GlobalStep { get; private set; } = 0;

    public GptCheckpointManager(string canonicalPath, string sidecarPath)
    {
        _canonicalPath = canonicalPath;
        _sidecarPath = sidecarPath;
    }

    public bool TryResume(GptNeuralFramework gpt, out string message)
    {
        if (!GptTrainingRunner.TryLoadModel(gpt, _canonicalPath, out message))
        {
            BestLoss = float.PositiveInfinity;
            GlobalStep = 0;
            return false;
        }

        LoadSidecar();
        message = $"[Checkpoint] Resumed. stepCount={gpt.StepCount} globalStep={GlobalStep} bestLoss={BestLoss:F4}";
        return true;
    }

    /// <summary>Always writes an epoch snapshot. Returns the path written.</summary>
    public string SaveEpochSnapshot(GptNeuralFramework gpt, int epoch, float loss)
    {
        string path = Path.Combine(
            Path.GetDirectoryName(_canonicalPath) ?? ".",
            $"shakespeare_gpt_epoch{epoch:D2}_loss{loss:F4}.bin");
        GptTrainingRunner.SaveModel(gpt, path);
        return path;
    }

    /// <summary>Writes the canonical checkpoint + sidecar only if this epoch is best.</summary>
    public bool TrySaveBest(GptNeuralFramework gpt, float loss, int globalStep)
    {
        if (loss >= BestLoss) return false;

        BestLoss = loss;
        GlobalStep = globalStep;
        GptTrainingRunner.SaveModel(gpt, _canonicalPath);
        SaveSidecar();
        return true;
    }

    public void NotifyGlobalStep(int step) => GlobalStep = step;

    private void SaveSidecar()
    {
        File.WriteAllLines(_sidecarPath, new[]
        {
            $"bestLoss={BestLoss.ToString(CultureInfo.InvariantCulture)}",
            $"globalStep={GlobalStep}"
        });
    }

    private void LoadSidecar()
    {
        if (!File.Exists(_sidecarPath)) return;

        foreach (var line in File.ReadAllLines(_sidecarPath))
        {
            var parts = line.Split('=');
            if (parts.Length != 2) continue;

            var key = parts[0].Trim();
            var val = parts[1].Trim();

            if (key == "bestLoss" &&
                float.TryParse(val, NumberStyles.Float, CultureInfo.InvariantCulture, out var bl))
                BestLoss = bl;
            else if (key == "globalStep" && int.TryParse(val, out var gs))
                GlobalStep = gs;
        }
    }
}
