using NeutralNET.Framework;

namespace NeutralTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int[] architecture = [2, 2, 1];

            var neuralFramework = new NeuralFramework(architecture);
            var gradientFramework = new NeuralFramework(architecture);

            neuralFramework.Run(gradientFramework);                     
        }
    }
}
