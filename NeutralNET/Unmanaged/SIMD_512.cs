using TElement = float;
using TVector = System.Runtime.Intrinsics.Vector512;
using FpVec = System.Runtime.Intrinsics.Vector512<float>;
using ByteVec = System.Runtime.Intrinsics.Vector512<byte>;
using TAvx = System.Runtime.Intrinsics.X86.Avx512F;
using Self = NeutralNET.SIMD_512;
using System.Runtime.Intrinsics;
using NeutralNET.Unmanaged;
using System.Numerics;
using System.Diagnostics;

namespace NeutralNET;

partial class SIMD_512
{
    public const int BitSize = 512;
    public const int ByteSize = BitSize / 8;
    public const int ElemCount = ByteSize / sizeof(TElement);
    public const int AlignSize = ElemCount;
    public const int AlignMask = AlignSize - 1;

    public const int ByteAlignSize = ByteSize;
    public const int ByteAlignMask = ByteAlignSize - 1;
}
partial class SIMD_512
{
    public static bool IsSupported { [MethodImpl(Inline)] get => TVector.IsHardwareAccelerated && TAvx.IsSupported; }
    [MethodImpl(Inline)]
    public static void EnsureSupported()
    {
        if (!IsSupported) throw new InvalidOperationException();
    }
    [Conditional("DEBUG"), MethodImpl(Inline)]
    public static void AssertSupported() => Debug.Assert(IsSupported);
}

#if true
partial class SIMDExtensions_512
{
    extension<T>(Self)
        where T : unmanaged
    {
        public static unsafe Vector512<T> LoadAligned(T* src) => TVector.LoadAligned(src);
        public static unsafe Vector512<T> LoadUnaligned(T* src) => TVector.LoadUnsafe(ref *src);
        public static unsafe Vector512<T> LoadUnaligned(ref T src) => TVector.LoadUnsafe(ref src);
    }
}
#endif
