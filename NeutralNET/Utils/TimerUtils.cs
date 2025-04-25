
using System.Diagnostics;

namespace NeutralNET.Utils;

public static class TimerUtils
{
    private static readonly Stopwatch _stopWatch = new Stopwatch();

    public static long PassedTime => _stopWatch.ElapsedMilliseconds;
    public static long Everything = 0;
    public static long TimeStamp => Stopwatch.GetTimestamp();
}
