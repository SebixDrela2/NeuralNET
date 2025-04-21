using NeutralNET.Framework;
using NeutralNET.Models;
using NeutralNET.Validators;

namespace NeutralTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            int[] architecture = [6, 10, 10, 6];
        
            var model = new BitModel();
            var neuralNetwork = new NeuralNetwork(architecture, model);
            var modelRunner = neuralNetwork.Run();

            var validator = new ThreeBitModelValidator(modelRunner);
            validator.Validate();
        }
    }
}
