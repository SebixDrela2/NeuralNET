using System;

namespace NeutralNET.GPU
{
    public static class GpuBackendFactory
    {
        private static IGpuBackend _instance;
        private static readonly object _lock = new object();
        private static bool _initialized = false;

        public static IGpuBackend Instance
        {
            get
            {
                if (!_initialized)
                {
                    lock (_lock)
                    {
                        if (!_initialized)
                        {
                            _instance = CreateBackend();
                            _initialized = true;
                        }
                    }
                }
                return _instance;
            }
        }

        private static IGpuBackend CreateBackend()
        {
            // Try TensorFlow first
            try
            {
                var tfBackend = new TensorFlowGpuBackend();
                if (tfBackend.IsAvailable)
                {
                    Console.WriteLine($"✅ Using TensorFlow backend: {tfBackend.DeviceName}");
                    return tfBackend;
                }
                tfBackend.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ TensorFlow unavailable: {ex.Message}");
            }

            // Fallback to CPU
            Console.WriteLine("⚠️ No GPU backend available. Using CPU.");
            return new CpuBackend();
        }

        public static bool IsGpuAvailable => Instance.IsAvailable;
    }
}
