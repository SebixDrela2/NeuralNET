namespace NeutralNET.Framework.Neural.CNN;

/// <summary>
/// Zero‑GC CNN framework with full object and buffer pooling, pluggable optimizers,
/// and low-latency P/Invoke CUDA/cuBLAS GPU matrix acceleration.
/// </summary>
///

internal record struct CnnSize(int BatchSize, int Channels, int Height, int Width);
