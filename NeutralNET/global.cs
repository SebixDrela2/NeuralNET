global using System.Runtime.CompilerServices;
// System.Runtime.CompilerServices
//  .MethodImplOptions
//  .MethodImplAttribute

global using static GlobalScope;
global using SIMD = NeutralNET.SIMD_512;
using System.Diagnostics.CodeAnalysis;

public static partial class GlobalScope
{
    public const MethodImplOptions Inline = MethodImplOptions.AggressiveInlining;
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
}
