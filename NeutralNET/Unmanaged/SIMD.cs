using System.Numerics;
using System.Runtime.Intrinsics;
using NeutralNET.Unmanaged;

namespace NeutralNET;

public static partial class SIMD_256;
public static partial class SIMD_512;
public static partial class SIMDExtensions_256;
public static partial class SIMDExtensions_512;
public static partial class VecExtensions;

partial class VecExtensions
{
    extension<T>(SIMD)
        where T : unmanaged, INumberBase<T>
    {
        [MethodImpl(Inline)] public static bool IsAligned(T offset) => (SIMD.ByteAlignMask & nuint.CreateTruncating(offset)) is 0;
        [MethodImpl(Inline)] public static void EnsureAligned(T offset) => UnalignedAddressException.ThrowIfNot(IsAligned(offset));
    }
    extension(SIMD)
    {
        [MethodImpl(Inline)] public static unsafe bool IsAligned(void* ptr) => IsAligned((nuint)ptr);
        [MethodImpl(Inline)] public static unsafe void EnsureAligned(void* ptr) => UnalignedAddressException.ThrowIfNot(IsAligned(ptr));
    }

    extension(UnalignedAddressException)
    {
        public static void ThrowIfNot(bool condition)
        {
            if (!condition) throw new UnalignedAddressException();
        }
        public static void ThrowIf(bool condition) => ThrowIfNot(!condition);
        public static void ThrowIfUnaligned(nuint value, nuint mask) => ThrowIfNotZero(value & mask);
    }
    extension<T>(UnalignedAddressException)
        where T : unmanaged, INumberBase<T>
    {
        public static void ThrowIfNotZero(T value) => ThrowIfNot(T.IsZero(value));
    }
}

public class UnalignedAddressException : Exception
{
    public UnalignedAddressException()
    { }
    public UnalignedAddressException(string? message)
        : base(message) { }
    public UnalignedAddressException(string? message, Exception? innerException)
        : base(message, innerException) { }
}
