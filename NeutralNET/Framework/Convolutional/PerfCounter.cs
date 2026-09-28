using System.Diagnostics;

namespace NeutralNET.Framework.Neural.CNN;

/// <summary>
/// Zero-GC CNN framework with full object and buffer pooling, pluggable optimizers,
/// and low-latency P/Invoke CUDA/cuBLAS GPU matrix acceleration.
/// </summary>
///

public sealed class PerfCounter
{
    private readonly Dictionary<string, long> _totalTicks = new();
    private readonly Dictionary<string, int> _calls = new();
    private readonly Stopwatch _stopwatch = Stopwatch.StartNew();
    private readonly string _name;
    private int _depth = 0;

    public PerfCounter(string name) { _name = name; }

    public IDisposable Measure(string label)
    {
        var start = _stopwatch.ElapsedTicks;
        return new Scope(this, label, start);
    }

    private void End(string label, long startTicks)
    {
        var elapsed = _stopwatch.ElapsedTicks - startTicks;

        _totalTicks.TryGetValue(label, out var currentTicks);
        _totalTicks[label] = currentTicks + elapsed;

        _calls.TryGetValue(label, out var currentCalls);
        _calls[label] = currentCalls + 1;
    }

    public void Reset()
    {
        _totalTicks.Clear();
        _calls.Clear();
    }

    public void Report(string tag = "")
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"=== PERF COUNTER [{_name}] {(tag == "" ? "" : "(" + tag + ")")} ===");

        long total = 0;
        foreach (var kv in _totalTicks) total += kv.Value;

        double msPerTick = 1000.0 / Stopwatch.Frequency;

        foreach (var kv in _totalTicks.OrderByDescending(k => k.Value))
        {
            long ticks = kv.Value;
            int calls = _calls[kv.Key];
            double ms = ticks * msPerTick;
            double pct = total > 0 ? 100.0 * ticks / total : 0;
            double msPerCall = calls > 0 ? ms / calls : 0;

            sb.AppendLine($"  {kv.Key,-40}  {ms,10:F2} ms  {pct,5:F1}%  calls={calls,8}  avg={msPerCall,8:F3} ms");
        }

        sb.AppendLine($"  {"TOTAL",-40}  {total * msPerTick,10:F2} ms");
        Console.WriteLine(sb.ToString());
    }

    private sealed class Scope : IDisposable
    {
        private readonly PerfCounter _parent;
        private readonly string _label;
        private readonly long _start;

        public Scope(PerfCounter parent, string label, long start)
        {
            _parent = parent;
            _label = label;
            _start = start;
        }

        public void Dispose() => _parent.End(_label, _start);
    }
}
