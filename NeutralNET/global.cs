global using System.Runtime.CompilerServices;
// System.Runtime.CompilerServices
//  .MethodImplOptions
//  .MethodImplAttribute

global using static GlobalScope;
global using SIMD = NeutralNET.SIMD_512;
using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.InteropServices;

public static partial class GlobalScope
{
    public const MethodImplOptions Inline = MethodImplOptions.AggressiveInlining;
    public static string ProjectDirectory => field ??= GetCallerDirectory();
    public static string SolutionDirectory => field ??= Path.GetFullPath(Path.GetDirectoryName(ProjectDirectory) ?? throw new InvalidOperationException());
    public static string BuildDirectory => field ??= Path.GetFullPath(Path.Join(SolutionDirectory, "Build"));

    [return: NotNullIfNotNull(nameof(target))]
    public static ref T DisposeReplace<T>([NotNullIfNotNull(nameof(value))] ref T target, T value)
    where T : IDisposable?, allows ref struct
    {
        Exchange(ref target, value)?.Dispose();
        return ref target;
    }

    [return: NotNullIfNotNull(nameof(target))]
    public static T Exchange<T>([NotNullIfNotNull(nameof(value))] scoped ref T target, T value)
    where T : allows ref struct
    {
        var prev = target;
        target = value;
        return prev;
    }

    public static string GetCallerDirectory([CallerFilePath] string path = "") => Path.GetFullPath(Path.GetDirectoryName(path) ?? throw new InvalidOperationException());
}

public static partial class Extensions;
partial class Extensions
{
    extension(NotSupportedException)
    {
        public static void ThrowIfFalse(
            [DoesNotReturnIf(false)] bool condition,
            [CallerArgumentExpression(nameof(condition))] string? expr = null,
            [CallerFilePath] string? origin = null,
            [CallerLineNumber] int ln = -1)
        {
            throw new NotSupportedException((expr, Path.GetFileName(origin), ln) switch
            {
                (null, _, _) => null,
                (var e, null, _) => $"{e} was false",
                var (e, name, line) => $"{e} was false (in {name}:{line})",
            });
        }
    }

    public ref struct ReadOnlyMinMaxResult<T>(ref readonly T min, ref readonly T max)
        where T : IBinaryNumber<T>
    {
        public ref readonly T Min = ref min;
        public ref readonly T Max = ref max;

        public ReadOnlyMinMaxResult(ref readonly T value) : this(in value, in value) { }
    }

    extension<T>(ReadOnlySpan<T> span)
    {
        public Y SumBy<Y>(Func<T, Y> selector)
            where Y : IBinaryNumber<Y>
        {
            Y sum = Y.Zero;
            foreach (ref readonly var x in span) sum += selector(x);
            return sum;
        }
    }
    extension<T>(ReadOnlySpan<T> span)
        where T : unmanaged
    {
        public int OffsetOf(ref readonly T element)
        {
            return !Unsafe.IsNullRef(in element)
                ? (int)(Unsafe.ByteOffset(in MemoryMarshal.GetReference(span), in element) / Unsafe.SizeOf<T>())
                : default;
        }
    }

    extension<T>(ReadOnlySpan<T> span)
        where T : IBinaryNumber<T>
    {
        public T Sum()
        {
            T sum = T.Zero;
            foreach (ref readonly var x in span) sum += x;
            return sum;
        }
        public ref readonly T Min()
        {
            if (span is []) return ref Unsafe.NullRef<T>();
            ref readonly T min = ref span[0];

            foreach (ref readonly var x in span)
            {
                if (x >= min) continue;
                min = ref x;
            }

            return ref min;
        }
        public ref readonly T Max()
        {
            if (span is []) return ref Unsafe.NullRef<T>();
            ref readonly T max = ref span[0];

            foreach (ref readonly var x in span)
            {
                if (x <= max) continue;
                max = ref x;
            }

            return ref max;
        }
        public ReadOnlyMinMaxResult<T> MinMax()
        {
            if (span is []) return default;
            ReadOnlyMinMaxResult<T> acc = new(in span[0]);

            foreach (ref readonly var x in span)
            {
                if (x < acc.Min) acc.Min = ref x;
                if (x > acc.Max) acc.Max = ref x;
            }

            return acc;
        }
    }

    extension<T>(List<T> xs)
        where T : IDisposable?
    {
        public void ClearAndDispose()
        {
            foreach (var x in xs) x?.Dispose();
            xs.Clear();
        }
    }

    extension<T>(T[] xs)
        where T : IDisposable?
    {
        public void DisposeEach()
        {
            foreach (var x in xs) x?.Dispose();
        }
    }

}
