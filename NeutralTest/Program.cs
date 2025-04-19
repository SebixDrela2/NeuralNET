using NeutralNET.Framework;
using NeutralNET.Models;

namespace NeutralTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int[] architecture = [8, 16, 8, 8];

            var neuralFramework = new NeuralFramework(architecture);
            var gradientFramework = new NeuralFramework(architecture);
            var model = new BitModel();

            neuralFramework.Run(gradientFramework, model);                     
        }
    }
}
