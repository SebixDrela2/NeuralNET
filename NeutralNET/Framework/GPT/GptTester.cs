using System;
using System.Diagnostics;
using System.IO;

namespace NeutralNET.Framework.Neural.GPT;

/// <summary>
/// Forward/backward benchmark and sanity harness. Independent of training.
/// </summary>
public class GpuTester
{
    private readonly GptNeuralFramework _gpt;
    private readonly GPTTextDataLoader _loader;
    private readonly TrainingConfig _cfg;

    public GpuTester(GptNeuralFramework gpt, GPTTextDataLoader loader, TrainingConfig cfg, string checkpointPath)
    {
        _gpt = gpt;
        _loader = loader;
        _cfg = cfg;

        if (File.Exists(checkpointPath))
        {
            try
            {
                GptTrainingRunner.LoadModel(gpt, checkpointPath);
                Console.WriteLine($"[Benchmark] Loaded weights from {checkpointPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Benchmark] Could not load checkpoint: {ex.Message}. Using random init.");
            }
        }
        else
        {
            Console.WriteLine("[Benchmark] No checkpoint found. Using random init.");
        }
    }

    public void Run()
    {
        PrintHardware();
        BenchmarkForward();
        BenchmarkBackward();
        PrintTopKLogits("To be or not to be");
        Console.WriteLine("\n[Benchmark] Done.");
    }

    private static void PrintHardware()
    {
        Console.WriteLine("\n=== CPU / SIMD support ===");
        Console.WriteLine($"AVX2     : {System.Runtime.Intrinsics.X86.Avx2.IsSupported}");
        Console.WriteLine($"AVX-512F : {System.Runtime.Intrinsics.X86.Avx512F.IsSupported}");
        Console.WriteLine($"FMA      : {System.Runtime.Intrinsics.X86.Fma.IsSupported}");
        Console.WriteLine($"Sse3     : {System.Runtime.Intrinsics.X86.Sse3.IsSupported}");
    }

    private void BenchmarkForward()
    {
        int seq = _cfg.ContextSize;
        int batch = _cfg.MaxBatchSize;

        Console.WriteLine($"\n=== Forward benchmark ({batch} x {seq} tokens) ===");

        int[] inputIds = new int[batch * seq];
        var rng = new Random(42);
        for (int i = 0; i < inputIds.Length; i++) inputIds[i] = rng.Next(_loader.VocabSize);

        unsafe
        {
            fixed (int* p = inputIds) _gpt.Forward(p, batch, seq);

            var sw = Stopwatch.StartNew();
            const int iters = 5;
            for (int i = 0; i < iters; i++)
                fixed (int* p = inputIds) _gpt.Forward(p, batch, seq);
            sw.Stop();

            double totalMs = sw.Elapsed.TotalMilliseconds;
            Console.WriteLine($"{iters} forwards: {totalMs:F2} ms total, {totalMs / iters:F2} ms/forward");
        }
    }

    private void BenchmarkBackward()
    {
        int seq = _cfg.ContextSize;
        int batch = _cfg.MaxBatchSize;
        int total = batch * seq;

        Console.WriteLine($"\n=== Fwd+Bwd benchmark ({batch} x {seq} tokens) ===");

        int[] inputIds = new int[total];
        int[] targets = new int[total];
        var rng = new Random(7);
        for (int i = 0; i < total; i++)
        {
            inputIds[i] = rng.Next(_loader.VocabSize);
            targets[i] = rng.Next(_loader.VocabSize);
        }

        unsafe
        {
            fixed (int* pi = inputIds) _gpt.Forward(pi, batch, seq);
            fixed (int* pt = targets) _gpt.Backward(pt, batch, seq);

            var sw = Stopwatch.StartNew();
            const int iters = 5;
            for (int i = 0; i < iters; i++)
            {
                fixed (int* pi = inputIds) _gpt.Forward(pi, batch, seq);
                fixed (int* pt = targets) _gpt.Backward(pt, batch, seq);
            }
            sw.Stop();

            double totalMs = sw.Elapsed.TotalMilliseconds;
            Console.WriteLine($"{iters} fwd+bwd: {totalMs:F2} ms total, {totalMs / iters:F2} ms/step");
        }
    }

    private unsafe void PrintTopKLogits(string prompt)
    {
        Console.WriteLine($"\n=== Top-5 logits for prompt \"{prompt}\" ===");

        int[] promptTokens = _loader.Encode(prompt).ToArray();
        fixed (int* p = promptTokens)
            _gpt.Forward(p, 1, promptTokens.Length);

        int last = promptTokens.Length - 1;
        float* logits = _gpt.LogitsOutput.Pointer + last * _gpt.LogitsOutput.ColumnsStride;
        int V = _gpt.VocabSize;

        Span<(int idx, float val)> top = stackalloc (int, float)[5];
        for (int i = 0; i < 5; i++) top[i] = (-1, float.NegativeInfinity);

        for (int v = 0; v < V; v++)
        {
            float val = logits[v];
            if (val <= top[4].val) continue;
            int pos = 4;
            while (pos > 0 && val > top[pos - 1].val) pos--;
            for (int k = 4; k > pos; k--) top[k] = top[k - 1];
            top[pos] = (v, val);
        }

        for (int i = 0; i < 5; i++)
        {
            char c = _loader.Decode(new[] { top[i].idx })[0];
            Console.WriteLine($"  #{i + 1}: id={top[i].idx,3} logit={top[i].val,8:F4}  char='{(c == '\n' ? '↵' : c)}'");
        }
    }
}
