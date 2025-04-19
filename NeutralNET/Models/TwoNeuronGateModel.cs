using NeutralNET.Stuff;
using System.Diagnostics.CodeAnalysis;

namespace NeutralNET.Models;

public class TwoNeuronGateModel
{
    private readonly (uint A, uint B, uint C)[] _trainingData;

    private readonly double _rate;
    private readonly double _epsillon;
    private readonly int _trainingLength;

    public TwoNeuronGateModel(
        (uint A, uint B, uint C)[] trainingData,
        double rate,
        double epsillion,
        int trainingLength)
    {
        _trainingData = trainingData;
        _rate = rate;
        _epsillon = epsillion;
        _trainingLength = trainingLength;
    }

    public void Run()
    {
        var xor = new XorModels();

        for (var i = 0; i < _trainingLength; i ++)
        {
            var gradient = GetGradient(xor);
            var loss = Loss(xor);

            Console.WriteLine($"Loss: {loss:f2} | W1: {xor.WeightOne:f2} | W2: {xor.WeightTwo:f2} | W3: {xor.WeightThree:f2} | W4: {xor.WeightFour:f2} | W5: {xor.WeightFive:f2} | W6: {xor.WeightSix:f2} | B1: {xor.BiasOne:f2} | B2: {xor.BiasTwo:f2} | B3: {xor.BiasThree:f2}");
            xor = Learn(xor, gradient);
        }

        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                xor.X = i;
                xor.Y = j;

                Console.WriteLine($"{i} ^ {j} = {Forward(xor):f10}");
            }
        }
    }

    private double Loss(XorModels xor)
    {
        double result = 0.0f;

        for (var i = 0; i < _trainingData.Length; i++)
        {
            xor.X = _trainingData[i].A;
            xor.Y = _trainingData[i].B;

            double forwardResult = Forward(xor);
            double distance = forwardResult - _trainingData[i].C;

            result += distance * distance;
        }

        result /= _trainingData.Length;
        return result;
    }

    private XorModels Learn(XorModels xor, XorModels gradient)
    {
        xor.WeightOne -= _rate * gradient.WeightOne;
        xor.WeightTwo -= _rate * gradient.WeightTwo;
        xor.BiasOne -= _rate * gradient.BiasOne;
        xor.WeightThree -= _rate * gradient.WeightThree;
        xor.WeightFour -= _rate * gradient.WeightFour;
        xor.BiasTwo -= _rate * gradient.BiasTwo;
        xor.WeightFive -= _rate * gradient.WeightFive;
        xor.WeightSix -= _rate * gradient.WeightSix;
        xor.BiasThree -= _rate * gradient.BiasThree;

        return xor;
    }

    private XorModels GetGradient(XorModels xor)
    {
        var gradient = new XorModels();
        var loss = Loss(xor);

        ModifyElementInternal(ref gradient.WeightOne, ref xor.WeightOne, xor, loss);      
        ModifyElementInternal(ref gradient.WeightTwo, ref xor.WeightTwo, xor, loss);      
        ModifyElementInternal(ref gradient.BiasOne, ref xor.BiasOne, xor, loss);        
        ModifyElementInternal(ref gradient.WeightThree, ref xor.WeightThree, xor, loss);      
        ModifyElementInternal(ref gradient.WeightFour, ref xor.WeightFour, xor, loss);        
        ModifyElementInternal(ref gradient.BiasTwo, ref xor.BiasTwo, xor, loss);      
        ModifyElementInternal(ref gradient.WeightFive, ref xor.WeightFive, xor, loss);       
        ModifyElementInternal(ref gradient.WeightSix, ref xor.WeightSix, xor, loss);        
        ModifyElementInternal(ref gradient.BiasThree, ref xor.BiasThree, xor, loss);       

        return gradient;
    }

    [SuppressMessage("Style", "IDE0059:Unnecessary assignment of a value", Justification = "<Pending>")]
    private void ModifyElementInternal(
        ref double gradientElement, 
        ref double element,
        XorModels xor,
        double loss)
    {
        var tmp = element;
        element += _epsillon;
        gradientElement = (Loss(xor) - loss) / _epsillon;
        element = tmp;
    }

    private static double Forward(XorModels xor)
    {
        var a = MathUtils.Sigmoid(xor.X * xor.WeightOne + xor.Y * xor.WeightTwo + xor.BiasOne);
        var b = MathUtils.Sigmoid(xor.X * xor.WeightThree + xor.Y * xor.WeightFour + xor.BiasTwo);

        return MathUtils.Sigmoid(a * xor.WeightFive + b * xor.WeightSix + xor.BiasThree);
    }
}
