using NeutralNET.GPU;

namespace NeutralNET.Framework.Convolutional.Native;
public record struct CublasContext : IDisposable
{
    private CublasItem<IntPtr> _cublasPtrs = default;
    private CublasItem<nuint> _cublasSizes = default;
    private CublasItem<int> _cublasStrides = default;
    private CublasTransitions _cublasTransitions = default;

    public CublasContext(
        CublasTransitions trans,
        CublasItem<int> size,
        CublasItem<int> stride)
    {
        int rowsA = (trans.A == CublasOperation.NonTranspose) ? size.A : size.C;
        int rowsB = (trans.B == CublasOperation.NonTranspose) ? size.C : size.B;
        int rowsC = size.A;

        nuint sizeA = (nuint)(rowsA * stride.A * sizeof(float));
        nuint sizeB = (nuint)(rowsB * stride.B * sizeof(float));
        nuint sizeC = (nuint)(rowsC * stride.C * sizeof(float));

        if (CudaInterop.cudaMalloc(out var d_A, sizeA) != 0 ||
            CudaInterop.cudaMalloc(out var d_B, sizeB) != 0 ||
            CudaInterop.cudaMalloc(out var d_C, sizeC) != 0)
        {
            throw new OutOfMemoryException("CUDA Memory Allocation failed.");
        }

        _cublasSizes.A = sizeA;
        _cublasSizes.B = sizeB;
        _cublasSizes.C = sizeC;

        _cublasPtrs.A = d_A;
        _cublasPtrs.B = d_B;
        _cublasPtrs.C = d_C;

        _cublasTransitions = trans;
        _cublasStrides = stride;
    }

    public CublasItem<int> GetStrides() => _cublasStrides;
    public CublasTransitions GetTransitions() => _cublasTransitions;

    public CublasItem<nuint> GetSizes() => _cublasSizes;

    public CublasItem<IntPtr> GetPointers() => _cublasPtrs;

    public void Dispose()
    {
        if (_cublasPtrs.A == IntPtr.Zero || _cublasPtrs.B == IntPtr.Zero || _cublasPtrs.C == IntPtr.Zero)
        {
            return;
        }

        CudaInterop.cudaFree(_cublasPtrs.A);
        CudaInterop.cudaFree(_cublasPtrs.B);
        CudaInterop.cudaFree(_cublasPtrs.C);

        _cublasPtrs.A = IntPtr.Zero;
        _cublasPtrs.B = IntPtr.Zero;
        _cublasPtrs.C = IntPtr.Zero;
    }
}
