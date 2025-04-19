using NeutralNET.Models;

namespace NeutralNET.Framework;

public class NeuralFramework
{
    public void Run()
    {
        var xorAdvanced = new XorAdvanced();
        xorAdvanced.Run(5, 5);
    }
}
