using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

namespace NeutralNET.Matrices;

[method: MethodImpl(Inline)]
[DebuggerDisplay($"{{{nameof(GetDebuggerDisplay)}(),nq}}")]
public unsafe struct AllocationHandle(void* ptr, nuint byteSize) : IDisposable
{
    public void* Pointer = ptr;
    public nuint ByteSize = byteSize;
    public readonly string SizeString => FormatSize(ByteSize);

    public readonly bool IsNull { [MethodImpl(Inline)] get => Pointer is null; }

    [MethodImpl(Inline)]
    public static AllocationHandle Alloc(nuint byteSize) => new(
        NativeMemory.AlignedAlloc(
            BitOperations.RoundUpToPowerOf2(byteSize),
            SIMD.ByteAlignSize
        ),
        byteSize
    );

    [MethodImpl(Inline)]
    public AllocationHandle Take() => Exchange(ref this, default);

    [MethodImpl(Inline)]
    public void Free()
    {
        if (Take() is not { IsNull: false } prev) return;
        NativeMemory.AlignedFree(prev.Pointer);
    }

    [MethodImpl(Inline)]
    public void Dispose()
    {
        if (Take() is not { IsNull: false } prev) return;
        NeuralMemoryPool.Return(prev);
    }

    public static string FormatSize(nuint size) => nuint.Log2(size) switch
    {
        >= 40 => $"{float.ScaleB(size, -40):f2} TB",
        >= 30 => $"{float.ScaleB(size, -30):f2} GB",
        >= 20 => $"{float.ScaleB(size, -20):f2} MB",
        >= 10 => $"{float.ScaleB(size, -10):f2} KB",
        < 10 => $"{size} B",
    };

    private readonly string GetDebuggerDisplay() => this switch
    {
        { IsNull: true } => "nullptr",
        { SizeString: var size } => size,
    };

    [Conditional("DEBUG")]
    public static void AssertSameSize(AllocationHandle lhs, AllocationHandle rhs, [CallerFilePath] string fp = "", [CallerLineNumber] int ln = 0)
    {
        if (lhs.ByteSize == rhs.ByteSize) return;

        const string fmtBg = "\e[48;2;46;7;9m";
        const string fmtWarn = $"{fmtBg}\e[K  \e[30;48;2;255;220;070m WARN {fmtBg}  \e[38;2;212;212;199;48;2;66;11;13m";

        var path = $"{Path.GetRelativePath(SolutionDirectory, fp)}:{ln}";

        Console.WriteLine($"{fmtWarn} Memory size mismatch between 0x{(nuint)lhs.Pointer:X016} \e[2m(size {lhs.SizeString,10})\e[22m and 0x{(nuint)rhs.Pointer:X016} \e[2m(size {rhs.SizeString,10})\e[22m {fmtBg} in \e[38;2;107;180;235m{path}\e[39;49m");
    }
}
