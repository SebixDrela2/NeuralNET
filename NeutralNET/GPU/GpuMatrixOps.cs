using System;
using System.Runtime.InteropServices;

namespace NeutralNET.GPU;

public static unsafe class CudaInterop
{
    private const string CudaRtDll = "cudart64_110.dll"; // Standard runtime DLL for CUDA 11.8

    public const int CudaMemcpyHostToDevice = 1;
    public const int CudaMemcpyDeviceToHost = 2;

    [DllImport(CudaRtDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int cudaMalloc(out IntPtr devPtr, nuint size);

    [DllImport(CudaRtDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int cudaFree(IntPtr devPtr);

    [DllImport(CudaRtDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern int cudaMemcpy(IntPtr dst, IntPtr src, nuint count, int kind);
}

public static unsafe class GpuMatrixOps
{
    private const string CublasDll = "cublas64_11.dll"; // cuBLAS DLL for CUDA 11.x

    public enum CublasOperation
    {
        NonTranspose = 0,
        Transpose = 1,
        ConjugateTranspose = 2
    }

    public enum CublasStatus
    {
        Success = 0,
        NotInitialized = 1,
        AllocFailed = 3,
        InvalidValue = 7,
        ArchMismatch = 8,
        MappingError = 11,
        ExecutionFailed = 13,
        InternalError = 14,
        NotSupported = 15,
        LicenseError = 16
    }

    [DllImport(CublasDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern CublasStatus cublasCreate_v2(out IntPtr handle);

    [DllImport(CublasDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern CublasStatus cublasDestroy_v2(IntPtr handle);

    [DllImport(CublasDll, CallingConvention = CallingConvention.Cdecl)]
    public static extern CublasStatus cublasSgemm_v2(
        IntPtr handle,
        CublasOperation transa,
        CublasOperation transb,
        int m,
        int n,
        int k,
        in float alpha,
        float* A,
        int lda,
        float* B,
        int ldb,
        in float beta,
        float* C,
        int ldc);

    private static IntPtr _cublasHandle;

    static GpuMatrixOps()
    {
        CublasStatus status = cublasCreate_v2(out _cublasHandle);
        if (status != CublasStatus.Success)
        {
            throw new Exception($"Failed to initialize cuBLAS handle. Status code: {status}");
        }
    }

    /// <summary>
    /// Executes Row-Major GEMM using cuBLAS Column-Major execution by staging CPU pointers through GPU Device Memory.
    /// Row-major: C [M x N] = op(A) [M x K] * op(B) [K x N]
    /// Equates to Column-major: C^T [N x M] = op(B)^T [N x K] * op(A)^T [K x M]
    /// </summary>
    private static void RowMajorSgemm(
        CublasOperation transA, CublasOperation transB,
        int m, int n, int k,
        float alpha,
        float* A, int strideA,
        float* B, int strideB,
        float beta,
        float* C, int strideC)
    {
        // Compute memory footprint based on actual row dimensions and strides
        int rowsA = (transA == CublasOperation.NonTranspose) ? m : k;
        int rowsB = (transB == CublasOperation.NonTranspose) ? k : n;
        int rowsC = m;

        nuint sizeA = (nuint)(rowsA * strideA * sizeof(float));
        nuint sizeB = (nuint)(rowsB * strideB * sizeof(float));
        nuint sizeC = (nuint)(rowsC * strideC * sizeof(float));

        IntPtr d_A = IntPtr.Zero;
        IntPtr d_B = IntPtr.Zero;
        IntPtr d_C = IntPtr.Zero;

        try
        {
            // 1. Allocate GPU Device Memory
            if (CudaInterop.cudaMalloc(out d_A, sizeA) != 0 ||
                CudaInterop.cudaMalloc(out d_B, sizeB) != 0 ||
                CudaInterop.cudaMalloc(out d_C, sizeC) != 0)
            {
                throw new OutOfMemoryException("CUDA Memory Allocation (cudaMalloc) failed.");
            }

            // 2. Transfer Host (CPU) -> Device (GPU)
            CudaInterop.cudaMemcpy(d_A, (IntPtr)A, sizeA, CudaInterop.CudaMemcpyHostToDevice);
            CudaInterop.cudaMemcpy(d_B, (IntPtr)B, sizeB, CudaInterop.CudaMemcpyHostToDevice);

            // 3. Execute cuBLAS SGEMM Kernel
            CublasStatus status = cublasSgemm_v2(
                _cublasHandle,
                transB, transA,
                n, m, k,
                in alpha,
                (float*)d_B, strideB,
                (float*)d_A, strideA,
                in beta,
                (float*)d_C, strideC);

            if (status != CublasStatus.Success)
            {
                throw new InvalidOperationException($"cuBLAS SGEMM execution failed with status code: {status}");
            }

            // 4. Transfer Device (GPU) -> Host (CPU)
            CudaInterop.cudaMemcpy((IntPtr)C, d_C, sizeC, CudaInterop.CudaMemcpyDeviceToHost);
        }
        finally
        {
            // 5. Free GPU Device Memory
            if (d_A != IntPtr.Zero) CudaInterop.cudaFree(d_A);
            if (d_B != IntPtr.Zero) CudaInterop.cudaFree(d_B);
            if (d_C != IntPtr.Zero) CudaInterop.cudaFree(d_C);
        }
    }

    public static void ComputeWeightGradientGpu(
        float* colInput,      // [patches x inDim]
        float* preGradMatrix, // [patches x filters]
        float* dW,            // [filters x inDim]
        int patches, int filters, int inDim,
        int strideA, int strideB, int strideC)
    {
        RowMajorSgemm(
            CublasOperation.Transpose, CublasOperation.NonTranspose,
            filters, inDim, patches,
            1.0f,
            preGradMatrix, strideB,
            colInput, strideA,
            0.0f,
            dW, strideC);
    }

    public static void ComputeGradientWithRespectToInputGpu(
        float* weightMat,     // [filters x inDim]
        float* preGradMatrix, // [patches x filters]
        float* gradPatchMat,  // [patches x inDim]
        int patches, int filters, int inDim,
        int strideWeight, int stridePreGrad, int strideGradPatch)
    {
        RowMajorSgemm(
            CublasOperation.NonTranspose, CublasOperation.NonTranspose,
            patches, inDim, filters,
            1.0f,
            preGradMatrix, stridePreGrad,
            weightMat, strideWeight,
            0.0f,
            gradPatchMat, strideGradPatch);
    }

    public static void ComputeConvolutionGpu(
        float* colInput,    // [patches x inDim]
        float* weightMat,   // [filters x inDim]
        float* result,      // [patches x filters]
        int patches, int filters, int inDim,
        int strideColInput, int strideWeight, int strideResult)
    {
        RowMajorSgemm(
            CublasOperation.NonTranspose, CublasOperation.Transpose,
            patches, filters, inDim,
            1.0f,
            colInput, strideColInput,
            weightMat, strideWeight,
            0.0f,
            result, strideResult);
    }

    public static void ComputeDenseForwardGpu(
        float* input,        // [batch x inFeatures]
        float* weights,      // [outFeatures x inFeatures]
        float* biases,
        float* result,       // [batch x outFeatures]
        int batch, int inFeatures, int outFeatures,
        int strideInput, int strideWeights, int strideResult)
    {
        RowMajorSgemm(
            CublasOperation.NonTranspose, CublasOperation.Transpose,
            batch, outFeatures, inFeatures,
            1.0f,
            input, strideInput,
            weights, strideWeights,
            0.0f,
            result, strideResult);

        // Vectorized bias addition
        for (int b = 0; b < batch; b++)
        {
            float* row = result + b * strideResult;
            for (int f = 0; f < outFeatures; f++)
            {
                row[f] += biases[f];
            }
        }
    }

    public static void ComputeDenseWeightGradientGpu(
        float* inputToLayer, // [batch x inDim]
        float* gradPre,      // [batch x outDim]
        float* dW,           // [inDim x outDim]
        int batch, int inDim, int outDim,
        int strideInput, int strideGradPre, int strideDW)
    {
        RowMajorSgemm(
            CublasOperation.Transpose, CublasOperation.NonTranspose,
            inDim, outDim, batch,
            1.0f,
            inputToLayer, strideInput,
            gradPre, strideGradPre,
            0.0f,
            dW, strideDW);
    }

    public static void ComputeDenseInputGradientGpu(
        float* gradPre,      // [batch x weightOutDim]
        float* weights,      // [weightOutDim x weightInDim]
        float* gradInput,    // [batch x weightInDim]
        int batch, int weightOutDim, int weightInDim,
        int strideGradPre, int strideWeights, int strideGradInput)
    {
        RowMajorSgemm(
            CublasOperation.NonTranspose, CublasOperation.NonTranspose,
            batch, weightInDim, weightOutDim,
            1.0f,
            gradPre, strideGradPre,
            weights, strideWeights,
            0.0f,
            gradInput, strideGradInput);
    }
}
