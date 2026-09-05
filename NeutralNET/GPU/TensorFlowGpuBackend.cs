using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NeutralNET.Activation;
using NeutralNET.Matrices;
using static NeutralNET.GPU.Native.TensorFlowNative;

namespace NeutralNET.GPU
{
    /// <summary>
    /// Encapsulates persistent GPU memory allocation via TensorFlow C API.
    /// </summary>
    public unsafe class GpuMatrixHandle : IDisposable
    {
        public TF_Tensor* TensorPointer { get; private set; }
        public int Rows { get; }
        public int Columns { get; }

        public GpuMatrixHandle(int rows, int cols)
        {
            Rows = rows;
            Columns = cols;
            long[] dims = new long[] { rows, cols };
            long byteSize = (long)rows * cols * sizeof(float);

            void* rawMemory = NativeMemory.AlignedAlloc((nuint)byteSize, 16);

            fixed (long* dimsPtr = dims)
            {
                var deallocatorPtr = (IntPtr)(delegate* unmanaged[Cdecl]<IntPtr, IntPtr, void>)&FreeAlignedDeallocator;
                TensorPointer = TF_NewTensor(
                    TF_DataType.TF_FLOAT,
                    dimsPtr,
                    2,
                    (IntPtr)rawMemory,
                    (UIntPtr)byteSize,
                    deallocatorPtr,
                    IntPtr.Zero
                );
            }
        }

        public float* GetBufferPointer() => (float*)TF_TensorData(TensorPointer);

        [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static void FreeAlignedDeallocator(IntPtr data, IntPtr arg)
        {
            if (data != IntPtr.Zero)
            {
                NativeMemory.AlignedFree((void*)data);
            }
        }

        public void Dispose()
        {
            if (TensorPointer != null)
            {
                TF_DeleteTensor(TensorPointer);
                TensorPointer = null;
            }
        }
    }

    public unsafe class TensorFlowGpuBackend : IGpuBackend
    {
        private bool _disposed;
        public bool IsAvailable { get; private set; }
        public string DeviceName { get; private set; }

        // REMOVED 'readonly' so it can be cleared/disposed properly
        private TF_Status* _status;
        private readonly ConcurrentDictionary<string, CachedSession> _sessionCache = new();

        private readonly ConditionalWeakTable<NeuralMatrix, GpuMatrixHandle> _gpuHandleMap = new();

        private class CachedSession
        {
            public TF_Graph* Graph;
            public TF_Session* Session;
            public TF_Output InputA;
            public TF_Output InputB;
            public TF_Output OutputMatMul;
        }

        static TensorFlowGpuBackend()
        {
            Environment.SetEnvironmentVariable("TF_CPP_MIN_LOG_LEVEL", "3");
            Environment.SetEnvironmentVariable("TF_ENABLE_ONEDNN_OPTS", "1");
        }

        public TensorFlowGpuBackend()
        {
            try
            {
                _status = TF_NewStatus();
                IsAvailable = true;
                DeviceName = "GPU (TensorFlow C API Async VRAM)";
            }
            catch
            {
                IsAvailable = false;
                DeviceName = "Unavailable";
            }
        }

        public void CopyToDevice(NeuralMatrix matrix)
        {
            if (!_gpuHandleMap.TryGetValue(matrix, out var handle) || handle.TensorPointer == null)
            {
                handle = new GpuMatrixHandle(matrix.Rows, matrix.UsedColumns);
                _gpuHandleMap.AddOrUpdate(matrix, handle);
            }

            float* dstBuffer = handle.GetBufferPointer();
            int rows = matrix.Rows;
            int cols = matrix.UsedColumns;
            int stride = matrix.ColumnsStride;

            for (int r = 0; r < rows; r++)
            {
                float* srcRow = matrix.Pointer + (r * stride);
                float* dstRow = dstBuffer + (r * cols);
                Unsafe.CopyBlockUnaligned(dstRow, srcRow, (uint)(cols * sizeof(float)));
            }
        }

        public void CopyToHost(NeuralMatrix matrix)
        {
            if (!_gpuHandleMap.TryGetValue(matrix, out var handle))
                return;

            float* srcBuffer = handle.GetBufferPointer();
            int rows = matrix.Rows;
            int cols = matrix.UsedColumns;
            int stride = matrix.ColumnsStride;

            for (int r = 0; r < rows; r++)
            {
                float* srcRow = srcBuffer + (r * cols);
                float* dstRow = matrix.Pointer + (r * stride);
                Unsafe.CopyBlockUnaligned(dstRow, srcRow, (uint)(cols * sizeof(float)));
            }
        }

        private GpuMatrixHandle GetOrAllocateGpuHandle(NeuralMatrix matrix)
        {
            if (!_gpuHandleMap.TryGetValue(matrix, out var handle))
            {
                handle = new GpuMatrixHandle(matrix.Rows, matrix.UsedColumns);
                _gpuHandleMap.Add(matrix, handle);
            }
            return handle;
        }

        public void Multiply(NeuralMatrix a, NeuralMatrix b, NeuralMatrix result)
        {
            if (!IsAvailable) throw new InvalidOperationException("GPU backend unavailable.");

            bool transposeA = false;
            bool transposeB = false;

            int m = a.Rows;
            int k = a.UsedColumns;
            int n = 0;

            if (b.Rows == k)
            {
                n = b.UsedColumns;
            }
            else if (b.UsedColumns == k)
            {
                transposeB = true;
                n = b.Rows;
            }
            else if (a.Rows == b.Rows)
            {
                transposeA = true;
                m = a.UsedColumns;
                k = a.Rows;
                n = b.UsedColumns;
            }
            else
            {
                throw new InvalidOperationException($"Incompatible shapes: A=[{a.Rows}x{a.UsedColumns}], B=[{b.Rows}x{b.UsedColumns}]");
            }

            GpuMatrixHandle handleA = GetOrAllocateGpuHandle(a);
            GpuMatrixHandle handleB = GetOrAllocateGpuHandle(b);
            GpuMatrixHandle handleResult = GetOrAllocateGpuHandle(result);

            string key = $"{a.Rows}_{a.UsedColumns}_{b.Rows}_{b.UsedColumns}_{transposeA}_{transposeB}";
            var cached = _sessionCache.GetOrAdd(key, _ => CreateCachedSession(a.Rows, a.UsedColumns, b.Rows, b.UsedColumns, transposeA, transposeB));

            TF_Output* inputs = stackalloc TF_Output[2] { cached.InputA, cached.InputB };
            TF_Tensor** inputValues = stackalloc TF_Tensor*[2] { handleA.TensorPointer, handleB.TensorPointer };

            TF_Output* outputs = stackalloc TF_Output[1] { cached.OutputMatMul };
            TF_Tensor** outputValues = stackalloc TF_Tensor*[1] { handleResult.TensorPointer };

            // Cast pointers to nint/IntPtr if native P/Invoke requires nint
            TF_SessionRun(
                (IntPtr)cached.Session,
                IntPtr.Zero,
                inputs,
                inputValues,
                2,
                outputs,
                outputValues,
                1,
                null,
                0,
                IntPtr.Zero,
                (IntPtr)_status
            );

            CheckStatus(_status, "GPU MatMul execution failed");
        }

        private CachedSession CreateCachedSession(int aRows, int aCols, int bRows, int bCols, bool transposeA, bool transposeB)
        {
            var graph = TF_NewGraph();
            long[] dimsA = new long[] { aRows, aCols };
            long[] dimsB = new long[] { bRows, bCols };

            fixed (long* dimsAPtr = dimsA)
            fixed (long* dimsBPtr = dimsB)
            {
                var descA = TF_NewOperation(graph, "Placeholder", "A");
                TF_SetAttrType(descA, "dtype", TF_DataType.TF_FLOAT);
                TF_SetAttrShape(descA, "shape", dimsAPtr, 2);
                var opA = TF_FinishOperation(descA, _status);
                CheckStatus(_status, "Placeholder A failed");

                var descB = TF_NewOperation(graph, "Placeholder", "B");
                TF_SetAttrType(descB, "dtype", TF_DataType.TF_FLOAT);
                TF_SetAttrShape(descB, "shape", dimsBPtr, 2);
                var opB = TF_FinishOperation(descB, _status);
                CheckStatus(_status, "Placeholder B failed");

                var descMatMul = TF_NewOperation(graph, "MatMul", "MatMul");
                TF_AddInput(descMatMul, new TF_Output { oper = opA, index = 0 });
                TF_AddInput(descMatMul, new TF_Output { oper = opB, index = 0 });
                TF_SetAttrBool(descMatMul, "transpose_a", transposeA);
                TF_SetAttrBool(descMatMul, "transpose_b", transposeB);
                var opMatMul = TF_FinishOperation(descMatMul, _status);
                CheckStatus(_status, "MatMul failed");

                var opts = TF_NewSessionOptions();
                var session = TF_NewSession(graph, opts, _status);
                TF_DeleteSessionOptions(opts);
                CheckStatus(_status, "Session creation failed");

                return new CachedSession
                {
                    Graph = graph,
                    Session = session,
                    InputA = new TF_Output { oper = opA, index = 0 },
                    InputB = new TF_Output { oper = opB, index = 0 },
                    OutputMatMul = new TF_Output { oper = opMatMul, index = 0 }
                };
            }
        }

        private void CheckStatus(TF_Status* status, string message)
        {
            if (TF_GetCode(status) != (int)TF_Code.TF_OK)
            {
                var msg = Marshal.PtrToStringAnsi(TF_Message(status));
                throw new InvalidOperationException($"{message}: {msg}");
            }
        }

        public NeuralMatrix Multiply(NeuralMatrix a, NeuralMatrix b) => throw new NotImplementedException("Use Multiply(a, b, result) overload.");
        public NeuralMatrix Add(NeuralMatrix a, NeuralMatrix b) => throw new NotImplementedException();
        public void Add(NeuralMatrix a, NeuralMatrix b, NeuralMatrix result) => throw new NotImplementedException();

        public NeuralMatrix Transpose(NeuralMatrix matrix)
        {
            var result = NeuralMatrix.GetOrCreate(matrix.UsedColumns, matrix.Rows);
            for (int i = 0; i < matrix.Rows; i++)
                for (int j = 0; j < matrix.UsedColumns; j++)
                    result.At(j, i) = matrix.At(i, j);
            return result;
        }

        public void ApplyActivation(NeuralMatrix matrix, ActivationType activation)
            => new ActivationSelector().GetActivation(activation)(matrix);

        public void Softmax(NeuralMatrix matrix) => ActivationFunctions.ApplySoftmaxVectorized(matrix);

        public void Synchronize() { }

        public void Dispose()
        {
            if (_disposed) return;

            foreach (var item in _sessionCache.Values)
            {
                if (item.Session != null) TF_DeleteSession(item.Session, _status);
                if (item.Graph != null) TF_DeleteGraph(item.Graph);
            }
            _sessionCache.Clear();

            if (_status != null)
            {
                TF_DeleteStatus(_status);
                _status = null;
            }

            _disposed = true;
        }
    }
}
