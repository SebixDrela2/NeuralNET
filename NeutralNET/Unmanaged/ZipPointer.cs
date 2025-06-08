using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace NeutralNET.Unmanaged;

internal readonly unsafe struct Zip2Pointer(float* aPtr, float* bPtr, float* endPtr)
{
    private readonly float* _a = aPtr;
    private readonly float* _b = bPtr;
    private readonly float* _end = endPtr;

    public float* A
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _a;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        init => _a = value;
    }

    public float* B
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _b;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        init => _b = value;
    }

    public bool IsInScope
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => _a < _end;
    }

    public Zip2Pointer(float* aPtr, float* bPtr, int length) 
        : this(aPtr, bPtr, aPtr + length) { }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Zip2Pointer operator ++(Zip2Pointer self) => self with { A = self.A + 1, B = self.B + 1 };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Zip2Pointer operator +(Zip2Pointer self, int value) => self with { A = self.A + value, B = self.B + value };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly void GetVectors(out Vector256<float> aVec, out Vector256<float> bVec)
    {
        aVec = Vector256.LoadAligned(A);
        bVec = Vector256.LoadAligned(B);
    }
}
