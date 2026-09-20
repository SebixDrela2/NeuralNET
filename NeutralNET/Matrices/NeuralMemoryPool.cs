using System.Collections.Concurrent;
using System.Numerics;

namespace NeutralNET.Matrices;

public static class NeuralMemoryPool
{
    public const int MaxAllowedSize_Pow2 = 28; // 0.25 GB
    public const ulong MaxAllowedSize = 1ul << MaxAllowedSize_Pow2;

    public static long CurrentCacheSize = 0;

    private static readonly ConcurrentBag<AllocationHandle>[] _pools = [.. Enumerable.Range(0, MaxAllowedSize_Pow2).Select(_ => new ConcurrentBag<AllocationHandle>())];

    [MethodImpl(Inline)]
    public static int BitWidth(ulong value) => 64 - BitOperations.LeadingZeroCount(value);

    [MethodImpl(Inline)]
    public static AllocationHandle RentBytes(nint size) => RentBytes((nuint)size);
    [MethodImpl(Inline)]
    public static AllocationHandle RentBytes(nuint size) => BitWidth(size) switch
    {
        <= 0 => throw new InvalidOperationException(),
        < MaxAllowedSize_Pow2 and var bw when TryTake(bw, out var x) => x,
        _ => AllocationHandle.Alloc(size),
    };

    private static bool TryTake(int bw, out AllocationHandle handle)
    {
        if (!_pools[bw].TryTake(out handle)) return false;
        Interlocked.Add(ref CurrentCacheSize, -(long)handle.ByteSize);
        return true;
    }

    [MethodImpl(Inline)]
    public static AllocationHandle Rent<T>(nint length) where T : unmanaged => Rent<T>((nuint)length);
    [MethodImpl(Inline)]
    public static AllocationHandle Rent<T>(nuint length) where T : unmanaged => RentBytes(length * (nuint)sizeof(T));

    [MethodImpl(Inline)]
    public static void Return(AllocationHandle handle)
    {
        if (handle.IsNull) return;

        switch (BitWidth(handle.ByteSize))
        {
            case <= 0: throw new InvalidOperationException();
            case >= MaxAllowedSize_Pow2:
                handle.Free();
                break;
            case var bw:
                _pools[bw].Add(handle);
                var newLen = (nuint)Interlocked.Add(ref CurrentCacheSize, (long)handle.ByteSize);
                if (nuint.Log2(newLen) > 30)
                {
                    Console.WriteLine($"\e[31;1mCache size is {newLen}\e[39m");
                }
                break;
        }
    }
}
