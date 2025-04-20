using NeutralNET.Framework;
using NeutralNET.Models;

namespace NeutralTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int[] architecture = [6, 4, 4, 6];

            var neuralFramework = new NeuralFramework(architecture);
            var gradientFramework = new NeuralFramework(architecture);
            var model = new BitModel();

            neuralFramework.Run(gradientFramework, model);                     
        }
    }
}
