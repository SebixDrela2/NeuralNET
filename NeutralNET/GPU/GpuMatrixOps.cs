using System;
using System.Runtime.InteropServices;

namespace NeutralNET.GPU
{
    public static unsafe class CudaInterop
    {
        private const string CudaRtDll = "cudart64_13.dll";

        public const int CudaMemcpyHostToDevice = 1;
        public const int CudaMemcpyDeviceToHost = 2;
        public const int CudaMemcpyDeviceToDevice = 3;

        [DllImport(CudaRtDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int cudaMalloc(out IntPtr devPtr, nuint size);

        [DllImport(CudaRtDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int cudaFree(IntPtr devPtr);

        [DllImport(CudaRtDll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int cudaMemcpy(IntPtr dst, IntPtr src, nuint count, int kind);
    }

    public static unsafe class GpuMatrixOps
    {
        private const string CublasDll = "cublas64_13.dll";

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
            int m, int n, int k,
            in float alpha,
            float* A, int lda,
            float* B, int ldb,
            in float beta,
            float* C, int ldc);

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
        /// Core GEMM operating directly on GPU Device pointers (eliminates PCIe round-trip overhead).
        /// </summary>
        public static void RowMajorSgemmDevice(
            CublasOperation transA, CublasOperation transB,
            int m, int n, int k,
            float alpha,
            IntPtr d_A, int strideA,
            IntPtr d_B, int strideB,
            float beta,
            IntPtr d_C, int strideC)
        {
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
        }

        /// <summary>
        /// Convenience wrapper for isolated host-to-host operations that require temporary staging.
        /// </summary>
        public static void RowMajorSgemmHostStaged(
            CublasOperation transA, CublasOperation transB,
            int m, int n, int k,
            float alpha,
            float* A, int strideA,
            float* B, int strideB,
            float beta,
            float* C, int strideC)
        {
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
                if (CudaInterop.cudaMalloc(out d_A, sizeA) != 0 ||
                    CudaInterop.cudaMalloc(out d_B, sizeB) != 0 ||
                    CudaInterop.cudaMalloc(out d_C, sizeC) != 0)
                {
                    throw new OutOfMemoryException("CUDA Memory Allocation failed.");
                }

                CudaInterop.cudaMemcpy(d_A, (IntPtr)A, sizeA, CudaInterop.CudaMemcpyHostToDevice);
                CudaInterop.cudaMemcpy(d_B, (IntPtr)B, sizeB, CudaInterop.CudaMemcpyHostToDevice);

                RowMajorSgemmDevice(transA, transB, m, n, k, alpha, d_A, strideA, d_B, strideB, beta, d_C, strideC);

                CudaInterop.cudaMemcpy((IntPtr)C, d_C, sizeC, CudaInterop.CudaMemcpyDeviceToHost);
            }
            finally
            {
                if (d_A != IntPtr.Zero) CudaInterop.cudaFree(d_A);
                if (d_B != IntPtr.Zero) CudaInterop.cudaFree(d_B);
                if (d_C != IntPtr.Zero) CudaInterop.cudaFree(d_C);
            }
        }
    }
}
