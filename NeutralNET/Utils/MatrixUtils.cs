using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NeutralNET.Utils;

public static class MatrixUtils
{
    public const int Alignment = 8;
    public const int AlignmentMask = Alignment - 1;

    public static int GetStride(int columns) => (columns + AlignmentMask) & ~AlignmentMask;

    public static uint[] GetStrideMask(int columns)
    {
        var strideMask = new uint[NeuralMatrix.Alignment];
        var computation = columns & AlignmentMask;
        computation = computation is 0 ? Alignment : computation;

        for (var i = 0; i < computation; ++i)
        {
            strideMask[i] = ~0u;
        }

        return strideMask;
    }
}
