using TElement = float;
using TVector = System.Runtime.Intrinsics.Vector256;
using FpVec = System.Runtime.Intrinsics.Vector256<float>;
using ByteVec = System.Runtime.Intrinsics.Vector256<byte>;
using TAvx = System.Runtime.Intrinsics.X86.Avx2;
using Self = NeutralNET.SIMD_256;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Diagnostics;
using NeutralNET.Unmanaged;
using System.Numerics;

namespace NeutralNET;

partial class SIMD_256
{
    public const int BitSize = 256;
    public const int ByteSize = BitSize / 8;
    public const int ElemCount = ByteSize / sizeof(TElement);
    public const int AlignSize = ElemCount;
    public const int AlignMask = AlignSize - 1;

    public const int ByteAlignSize = ByteSize;
    public const int ByteAlignMask = ByteAlignSize - 1;
}

partial class SIMD_256
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

#if false
partial class SIMDExtensions_256
{
    extension<T>(Self)
        where T : unmanaged
    {
        public static unsafe Vector256<T> LoadAligned(T* src) => TVector.LoadAligned(src);
        public static unsafe Vector256<T> LoadUnaligned(T* src) => TVector.LoadUnsafe(ref *src);
        public static unsafe Vector256<T> LoadUnaligned(ref T src) => TVector.LoadUnsafe(ref src);
    }
}
#endif
