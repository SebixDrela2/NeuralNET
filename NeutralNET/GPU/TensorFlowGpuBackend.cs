using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NeutralNET.Activation;
using NeutralNET.Matrices;
using static NeutralNET.GPU.Native.TensorFlowNative;

namespace NeutralNET.GPU
{
    public unsafe class TensorFlowGpuBackend : IGpuBackend
    {
        private bool _disposed = false;

        public bool IsAvailable { get; private set; }
        public string DeviceName { get; private set; }

        public TensorFlowGpuBackend()
        {
            
            try
            {
                Environment.SetEnvironmentVariable("TF_CPP_MIN_LOG_LEVEL", "2");
                // Test session creation
                var testStatus = TF_NewStatus();
                var testGraph = TF_NewGraph();
                var opts = TF_NewSessionOptions();
                var testSession = TF_NewSession(testGraph, opts, testStatus);
                var code = TF_GetCode(testStatus);
                if (code != (int)TF_Code.TF_OK)
                {
                    var msg = Marshal.PtrToStringAnsi(TF_Message(testStatus));
                    throw new Exception($"TF_NewSession failed: {msg}");
                }
                TF_DeleteSession(testSession, testStatus);
                TF_DeleteGraph(testGraph);
                TF_DeleteSessionOptions(opts);
                TF_DeleteStatus(testStatus);

                IsAvailable = true;
                DeviceName = "GPU (TensorFlow C API)";
                
            }
            catch (Exception ex)
            {
                
                IsAvailable = false;
                DeviceName = "Unavailable";
            }
            
        }

        // --------------------------------------------------------------------
        // Multiply with pre-allocated result
        // --------------------------------------------------------------------
        public void Multiply(NeuralMatrix a, NeuralMatrix b, NeuralMatrix result)
        {
            if (!IsAvailable)
                throw new InvalidOperationException("GPU not available");

            

            bool transposeB;
            int expectedRows, expectedCols;
            if (a.UsedColumns == b.Rows)
            {
                transposeB = false;
                expectedRows = a.Rows;
                expectedCols = b.UsedColumns;
                
            }
            else if (a.UsedColumns == b.UsedColumns)
            {
                transposeB = true;
                expectedRows = a.Rows;
                expectedCols = b.Rows;
                
            }
            else
            {
                throw new InvalidOperationException($"Incompatible dims: A={a.Rows}x{a.UsedColumns}, B={b.Rows}x{b.UsedColumns}");
            }

            if (result.Rows != expectedRows || result.UsedColumns != expectedCols)
                throw new InvalidOperationException($"Result shape mismatch: expected [{expectedRows}, {expectedCols}], got [{result.Rows}, {result.UsedColumns}]");

            var status = TF_NewStatus();
            var graph = TF_NewGraph();
            TF_Session* session = null;
            TF_Tensor* tensorA = null;
            TF_Tensor* tensorB = null;

            long[] dimsA = new long[] { a.Rows, a.UsedColumns };
            long[] dimsB = new long[] { b.Rows, b.UsedColumns };

            try
            {
                fixed (long* dimsAPtr = dimsA)
                fixed (long* dimsBPtr = dimsB)
                {
                    // Placeholder A with explicit shape
                    var descA = TF_NewOperation(graph, "Placeholder", "A");
                    TF_SetAttrType(descA, "dtype", TF_DataType.TF_FLOAT);
                    TF_SetAttrShape(descA, "shape", dimsAPtr, 2);
                    var opA = TF_FinishOperation(descA, status);
                    CheckStatus(status, "Placeholder A failed", graph);

                    TF_Output inputA = new TF_Output { oper = opA, index = 0 };

                    // Placeholder B with explicit shape
                    var descB = TF_NewOperation(graph, "Placeholder", "B");
                    TF_SetAttrType(descB, "dtype", TF_DataType.TF_FLOAT);
                    TF_SetAttrShape(descB, "shape", dimsBPtr, 2);
                    var opB = TF_FinishOperation(descB, status);
                    CheckStatus(status, "Placeholder B failed", graph);

                    TF_Output inputB = new TF_Output { oper = opB, index = 0 };

                    // MatMul Operation
                    var descMatMul = TF_NewOperation(graph, "MatMul", "MatMul");
                    TF_AddInput(descMatMul, inputA);
                    TF_AddInput(descMatMul, inputB);
                    TF_SetAttrBool(descMatMul, "transpose_b", transposeB);
                    var opMatMul = TF_FinishOperation(descMatMul, status);
                    CheckStatus(status, "MatMul failed", graph);

                    TF_Output outputOp = new TF_Output { oper = opMatMul, index = 0 };

                    // Session creation
                    var opts = TF_NewSessionOptions();
                    session = TF_NewSession(graph, opts, status);
                    TF_DeleteSessionOptions(opts);
                    CheckStatus(status, "Session creation failed", graph);

                    // Prepare input Tensors
                    tensorA = CreateAlignedTensor(a.Pointer, a.Rows, a.UsedColumns);
                    tensorB = CreateAlignedTensor(b.Pointer, b.Rows, b.UsedColumns);

                    // Setup fixed-size arrays on stack for C API consumption
                    TF_Output* inputs = stackalloc TF_Output[2] { inputA, inputB };
                    TF_Tensor** inputValues = stackalloc TF_Tensor*[2] { tensorA, tensorB };

                    TF_Output* outputs = stackalloc TF_Output[1] { outputOp };
                    TF_Tensor** outputValues = stackalloc TF_Tensor*[1];
                    outputValues[0] = null;

                    TF_SessionRun(
                        (IntPtr)session,
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
                        (IntPtr)status
                    );

                    CheckStatus(status, "TF_SessionRun failed", graph, session);

                    TF_Tensor* resultTensor = outputValues[0];
                    if (resultTensor == null)
                        throw new Exception("resultTensor is null after TF_SessionRun");

                    CopyTensorToPointer(resultTensor, result.Pointer, expectedRows, expectedCols);

                    TF_DeleteTensor(resultTensor);
                    
                }
            }
            finally
            {
                if (tensorA != null) TF_DeleteTensor(tensorA);
                if (tensorB != null) TF_DeleteTensor(tensorB);
                if (session != null) TF_DeleteSession(session, status);
                if (graph != null) TF_DeleteGraph(graph);
                if (status != null) TF_DeleteStatus(status);
            }
        }

        public NeuralMatrix Multiply(NeuralMatrix a, NeuralMatrix b)
        {
            throw new NotImplementedException("Use Multiply with result overload");
        }

        private void CheckStatus(TF_Status* status, string message, TF_Graph* graph = null, TF_Session* session = null)
        {
            if (TF_GetCode(status) != (int)TF_Code.TF_OK)
            {
                var msg = Marshal.PtrToStringAnsi(TF_Message(status));
                if (session != null) TF_DeleteSession(session, status);
                if (graph != null) TF_DeleteGraph(graph);
                if (status != null) TF_DeleteStatus(status);
                throw new Exception($"{message}: {msg}");
            }
        }

        // --------------------------------------------------------------------
        // Aligned tensor creation (16-byte aligned for AVX)
        // --------------------------------------------------------------------
        private TF_Tensor* CreateAlignedTensor(float* data, int rows, int cols)
        {
            long byteSize = (long)rows * cols * sizeof(float);
            var alignedData = (IntPtr)NativeMemory.AlignedAlloc((nuint)byteSize, 16);
            Buffer.MemoryCopy(data, (float*)alignedData, byteSize, byteSize);

            long[] dims = new long[] { rows, cols };
            fixed (long* dimsPtr = dims)
            {
                var deallocatorPtr = (IntPtr)(delegate* unmanaged[Cdecl]<IntPtr, IntPtr, void>)&FreeAlignedDeallocator;
                return TF_NewTensor(
                    TF_DataType.TF_FLOAT,
                    dimsPtr,
                    2,
                    alignedData,
                    (UIntPtr)byteSize,
                    deallocatorPtr,
                    IntPtr.Zero
                );
            }
        }

        [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static void FreeAlignedDeallocator(IntPtr data, IntPtr arg)
        {
            NativeMemory.AlignedFree((void*)data);
        }

        private void CopyTensorToPointer(TF_Tensor* tensor, float* dst, int rows, int cols)
        {
            var src = (float*)TF_TensorData(tensor);
            long totalBytes = (long)rows * cols * sizeof(float);
            Buffer.MemoryCopy(src, dst, totalBytes, totalBytes);
        }

        // --------------------------------------------------------------------
        // Other methods
        // --------------------------------------------------------------------
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
        public void CopyToDevice(NeuralMatrix matrix) { }
        public void CopyToHost(NeuralMatrix matrix) { }
        public void Synchronize() { }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
        }
    }
}
