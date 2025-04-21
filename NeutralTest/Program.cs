using NeutralNET.Framework;
using NeutralNET.Models;

namespace NeutralTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int[] architecture = [6, 4, 4, 6];
        
            var model = new BitModel();
            var neuralNetwork = new NeuralNetwork(architecture, model);
            neuralNetwork.Run();                     
        }
    }
}
