using System;
using System.Runtime.InteropServices;

namespace NeutralNET.GPU.Native
{
    public static unsafe class TensorFlowNative
    {
        private const string DllName = "tensorflow.dll"; // Windows
        // For Linux: "libtensorflow.so"
        // For Mac:   "libtensorflow.dylib"

        // ========================================================================
        // Status
        // ========================================================================
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TF_Status* TF_NewStatus();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TF_DeleteStatus(TF_Status* status);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int TF_GetCode(TF_Status* status);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr TF_Message(TF_Status* status);

        // ========================================================================
        // Session & Graph
        // ========================================================================
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TF_Graph* TF_NewGraph();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TF_DeleteGraph(TF_Graph* graph);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TF_SessionOptions* TF_NewSessionOptions();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TF_DeleteSessionOptions(TF_SessionOptions* opts);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TF_SetConfig(TF_SessionOptions* opts, byte[] proto, ulong proto_len, TF_Status* status);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TF_Session* TF_NewSession(TF_Graph* graph, TF_SessionOptions* opts, TF_Status* status);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TF_DeleteSession(TF_Session* session, TF_Status* status);

        // ========================================================================
        // Tensor
        // ========================================================================
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TF_Tensor* TF_NewTensor(
            TF_DataType dtype,
            long* dims,
            int num_dims,
            IntPtr data,
            UIntPtr len,
            IntPtr deallocator,
            IntPtr deallocator_arg
        );

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TF_DeleteTensor(TF_Tensor* tensor);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr TF_TensorData(TF_Tensor* tensor);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern long TF_TensorByteSize(TF_Tensor* tensor);

        // ========================================================================
        // Operations
        // ========================================================================
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TF_OperationDescription* TF_NewOperation(
            TF_Graph* graph,
            string op_type,
            string op_name
        );

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TF_DeleteOperationDescription(TF_OperationDescription* desc);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TF_AddInput(TF_OperationDescription* desc, TF_Output input);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TF_SetAttrBool(TF_OperationDescription* desc, string attr_name, bool value);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TF_SetAttrType(TF_OperationDescription* desc, string attr_name, TF_DataType value);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TF_Operation* TF_FinishOperation(TF_OperationDescription* desc, TF_Status* status);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern TF_Operation* TF_GraphOperationByName(TF_Graph* graph, string name);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static unsafe extern void TF_SessionRun(
            IntPtr session,
            IntPtr runOptions,
            TF_Output* inputs,
            TF_Tensor** inputValues,
            int ninputs,
            TF_Output* outputs,
            TF_Tensor** outputValues,
            int noutputs,
            TF_Operation** targetOpers,
            int ntargets,
            IntPtr runMetadata,
            IntPtr status
        );

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void TF_SetAttrShape(
            TF_OperationDescription* desc,
            string attr_name,
            long* dims,
            int num_dims
        );

        // ========================================================================
        // Types
        // ========================================================================
        public enum TF_DataType : int
        {
            TF_FLOAT = 1,
            TF_DOUBLE = 2,
            TF_INT32 = 3,
            TF_UINT8 = 4,
            TF_INT16 = 5,
            TF_INT8 = 6,
            TF_STRING = 7,
            TF_COMPLEX64 = 8,
            TF_INT64 = 9,
            TF_BOOL = 10,
            TF_QINT8 = 11,
            TF_QUINT8 = 12,
            TF_QINT32 = 13,
            TF_BFLOAT16 = 14,
            TF_QINT16 = 15,
            TF_QUINT16 = 16,
            TF_UINT16 = 17,
            TF_COMPLEX128 = 18,
            TF_HALF = 19,
            TF_RESOURCE = 20,
            TF_VARIANT = 21,
            TF_UINT32 = 22,
            TF_UINT64 = 23,
        }

        public enum TF_Code : int
        {
            TF_OK = 0,
            TF_CANCELLED = 1,
            TF_UNKNOWN = 2,
            TF_INVALID_ARGUMENT = 3,
            TF_DEADLINE_EXCEEDED = 4,
            TF_NOT_FOUND = 5,
            TF_ALREADY_EXISTS = 6,
            TF_PERMISSION_DENIED = 7,
            TF_UNAUTHENTICATED = 16,
            TF_RESOURCE_EXHAUSTED = 8,
            TF_FAILED_PRECONDITION = 9,
            TF_ABORTED = 10,
            TF_OUT_OF_RANGE = 11,
            TF_UNIMPLEMENTED = 12,
            TF_INTERNAL = 13,
            TF_UNAVAILABLE = 14,
            TF_DATA_LOSS = 15,
        }

        // ========================================================================
        // Structs
        // ========================================================================
        [StructLayout(LayoutKind.Sequential)]
        public struct TF_Status { }

        [StructLayout(LayoutKind.Sequential)]
        public struct TF_Graph { }

        [StructLayout(LayoutKind.Sequential)]
        public struct TF_SessionOptions { }

        [StructLayout(LayoutKind.Sequential)]
        public struct TF_Session { }

        [StructLayout(LayoutKind.Sequential)]
        public struct TF_Tensor { }

        [StructLayout(LayoutKind.Sequential)]
        public struct TF_Operation { }

        [StructLayout(LayoutKind.Sequential)]
        public struct TF_OperationDescription { }

        [StructLayout(LayoutKind.Sequential)]
        public struct TF_Output
        {
            public TF_Operation* oper;
            public int index;
        }
    }
}
