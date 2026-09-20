using System.Diagnostics;

namespace NeutralNET.Framework.Convolutional;

public class SourceLocation(
    MatrixInfo info,
    [CallerFilePath] string fp = "",
    [CallerLineNumber] int ln = 0)
{
    public static SourceLocation Current(
        MatrixInfo info,
        [CallerFilePath] string fp = "",
        [CallerLineNumber] int ln = 0
    ) => new SourceLocation(info, fp, ln);
    public StackTrace Trace { get; } = new StackTrace();

    public int ThreadID = Environment.CurrentManagedThreadId;
    public long TimeStamp = Stopwatch.GetTimestamp();

    public MatrixInfo Info { get; } = info;
    public int LineNumber { get; } = ln;
    public string FilePath { get; } = fp[53..];

    public string Debug => $"[{TimeStamp}|{ThreadID}] {FilePath}:{LineNumber}\n{Trace}";
    public override string ToString() => Debug;
}
