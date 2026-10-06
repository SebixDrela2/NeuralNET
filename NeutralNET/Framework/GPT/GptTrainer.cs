using System.Diagnostics;

namespace NeutralNET.Framework.Neural.GPT;

/// <summary>
/// Owns the training loop: LR schedule, per-batch logging, epoch evaluation,
/// checkpointing. Delegates checkpointing to <see cref="GptCheckpointManager"/>.
/// </summary>
public class GptTrainer
{
    private readonly GptNeuralFramework _gpt;
    private readonly GptCheckpointManager _checkpoints;
    private readonly TrainingConfig _cfg;
    private readonly List<int[][]> _miniBatches;

    public GptTrainer(
        GptNeuralFramework gpt,
        GptCheckpointManager checkpoints,
        TrainingConfig cfg,
        List<int[][]> miniBatches)
    {
        _gpt = gpt;
        _checkpoints = checkpoints;
        _cfg = cfg;
        _miniBatches = miniBatches;
    }

    public void Train()
    {
        int batchesPerEpoch = _miniBatches.Count;
        int totalSteps = batchesPerEpoch * _cfg.TotalEpochs;
        float baseLr = _cfg.LearningRate;
        int globalStep = _checkpoints.GlobalStep;
        float bestLoss = _checkpoints.BestLoss;

        for (int epoch = 1; epoch <= _cfg.TotalEpochs; epoch++)
        {
            Console.WriteLine($"\n--- Epoch {epoch:D2}/{_cfg.TotalEpochs:D2} ---");

            var epochTimer = Stopwatch.StartNew();
            int batchIndex = 0;

            foreach (var miniBatch in _miniBatches)
            {
                globalStep++;
                _gpt.SetLearningRate(ComputeLr(globalStep, totalSteps, baseLr));

                var batchTimer = Stopwatch.StartNew();
                _gpt.TrainStep(miniBatch);
                batchTimer.Stop();
                batchIndex++;

                if (GptNeuralFramework.DiagnosticsEnabled && batchIndex % _cfg.BatchDiagEvery == 0)
                    PrintBatchDiag();

                if (batchIndex == 1 || (batchIndex & 15) == 0 || batchIndex == batchesPerEpoch)
                    PrintProgress(epochTimer, batchTimer, batchIndex, batchesPerEpoch);
            }

            var (avgLoss, accuracy) = GptTrainingRunner.Evaluate(_gpt, _miniBatches);
            Console.WriteLine(
                $"Epoch {epoch:D2} Finished | Loss: {avgLoss:F4} | Accuracy: {accuracy:F2}% | " +
                $"Total Epoch Time: {epochTimer.Elapsed.TotalSeconds:F2}s");

            SaveSnapshots(epoch, avgLoss, globalStep);
        }
    }

    private float ComputeLr(int globalStep, int totalSteps, float baseLr)
    {
        float warmup = MathF.Min(1f, (float)globalStep / _cfg.WarmupSteps);
        float progress = MathF.Min(1f, (float)globalStep / totalSteps);
        float cosine = 0.5f * (1f + MathF.Cos(MathF.PI * progress));
        float floor = _cfg.FinalLrMultiplier;
        float decay = floor + (1f - floor) * cosine;
        return baseLr * warmup * decay;
    }

    private void SaveSnapshots(int epoch, float loss, int globalStep)
    {
        try
        {
            string path = _checkpoints.SaveEpochSnapshot(_gpt, epoch, loss);
            Console.WriteLine($"[Checkpoint] Epoch snapshot saved: {path}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Checkpoint] Epoch snapshot save failed: {ex.Message}");
        }

        if (_checkpoints.TrySaveBest(_gpt, loss, globalStep))
        {
            Console.WriteLine($"[Best] New best model saved (loss {loss:F4}) → {_cfg.CanonicalCheckpoint}");
        }
        else
        {
            Console.WriteLine($"[Best] Not better than previous best ({_checkpoints.BestLoss:F4}). Canonical checkpoint unchanged.");
        }
    }

    private void PrintProgress(Stopwatch epochTimer, Stopwatch batchTimer, int batchIndex, int totalBatches)
    {
        float progress = (float)batchIndex / totalBatches * 100f;
        double batchMs = batchTimer.Elapsed.TotalMilliseconds;
        double elapsedSec = epochTimer.Elapsed.TotalSeconds;
        double itemsPerSec = batchIndex / elapsedSec;

        Console.WriteLine(
            $"Batch {batchIndex}/{totalBatches} [{progress:F1}%] - Step Time: {batchMs:F2} ms | " +
            $"LR: {_gpt.Config.LearningRate:E2} | Avg Speed: {itemsPerSec:F1} batch/s");
    }

    private unsafe void PrintBatchDiag()
    {
        float* q = _gpt.Layers[0].Wq.Pointer;
        float* out2 = _gpt.OutputProjection.Pointer;
        float* tok = _gpt.TokenEmbeddings.Pointer;
        float* logits = _gpt.LogitsOutput.Pointer;

        int qCount = _gpt.Layers[0].Wq.Rows * _gpt.Layers[0].Wq.ColumnsStride;
        int outCount = _gpt.OutputProjection.Rows * _gpt.OutputProjection.ColumnsStride;
        int tokCount = _gpt.TokenEmbeddings.Rows * _gpt.TokenEmbeddings.ColumnsStride;
        int logCount = _gpt.LogitsOutput.Rows * _gpt.LogitsOutput.ColumnsStride;

        float qMax = 0f, outMax = 0f, tokMax = 0f, logMax = 0f;
        for (int i = 0; i < qCount; i++) if (MathF.Abs(q[i]) > qMax) qMax = MathF.Abs(q[i]);
        for (int i = 0; i < outCount; i++) if (MathF.Abs(out2[i]) > outMax) outMax = MathF.Abs(out2[i]);
        for (int i = 0; i < tokCount; i++) if (MathF.Abs(tok[i]) > tokMax) tokMax = MathF.Abs(tok[i]);
        for (int i = 0; i < logCount; i++) if (MathF.Abs(logits[i]) > logMax) logMax = MathF.Abs(logits[i]);

        Console.WriteLine(
            $"[batch-diag] |Wq|max={qMax:G4} |Wo|max={outMax:G4} |tok|max={tokMax:G4} " +
            $"|logits|max={logMax:G4}");
    }
}
