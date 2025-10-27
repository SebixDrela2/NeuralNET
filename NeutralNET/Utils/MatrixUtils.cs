using NeutralNET.Matrices;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace NeutralNET.Utils;

public static class MatrixUtils
{
    public const int AlignmentMask = NeuralMatrix.Alignment - 1;

    public static int GetStride(int columns) => (columns + AlignmentMask) & ~AlignmentMask;

    public static uint[] GetStrideMask(int columns)
    {
        var strideMask = new uint[NeuralMatrix.Alignment];
        var computation = columns & AlignmentMask;
        computation = computation is 0 ? NeuralMatrix.Alignment : computation;

        for (var i = 0; i < computation; ++i)
        {
            strideMask[i] = ~0u;
        }

        return strideMask;
    }
}
