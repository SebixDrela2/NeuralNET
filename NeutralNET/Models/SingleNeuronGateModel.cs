using NeutralNET.Stuff;

namespace NeutralNET.Models;

public class SingleNeuronGateModel
{
    private readonly (uint A, uint B, uint C)[] _trainingData;

    private readonly double _rate;
    private readonly double _epsillon;
    private readonly int _trainingLength;

    public SingleNeuronGateModel(
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
        var weightOne = RandomUtils.GetDouble(1);
        var weightTwo = RandomUtils.GetDouble(1);

        var bias = RandomUtils.GetDouble(1);

        for (var index = 0; index < _trainingLength; index++)
        {
            var (dlossOne, dLossTwo) = DLoss(weightOne, weightTwo, bias);
            var dBias = DBias(weightOne, weightTwo, bias);

            weightOne -= _rate * dlossOne;
            weightTwo -= _rate * dLossTwo;
            bias -= _rate * dBias;

            var loss = Loss(weightOne, weightTwo, bias);

            (double A, double B, double C, double D) tmp = (
                1 - Math.Abs((MathUtils.Sigmoid(0 * weightOne + 0 * weightTwo + bias) - _trainingData[0].C)),
                1 - Math.Abs((MathUtils.Sigmoid(0 * weightOne + 1 * weightTwo + bias) - _trainingData[1].C)),
                1 - Math.Abs((MathUtils.Sigmoid(1 * weightOne + 0 * weightTwo + bias) - _trainingData[2].C)),
                1 - Math.Abs((MathUtils.Sigmoid(1 * weightOne + 1 * weightTwo + bias) - _trainingData[3].C))
            );

            Console.Write($"{weightOne,5:f2} | {weightTwo,5:f2} | {loss,5:f5} | {bias,5:f2} || ");
            Console.WriteLine($"{tmp.A,5:P} | {tmp.B,5:P} | {tmp.C,5:P} | {tmp.D,5:P} | ");
        }     

        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                Console.WriteLine($"{i} | {j} = {MathUtils.Sigmoid(i * weightOne + j * weightTwo + bias):f10}");
            }
        }
    }

    private (double, double) DLoss(double weightOne, double weightTwo, double bias)
    {
        var dloss = Loss(weightOne, weightTwo, bias);

        var dlossOne = (Loss(weightOne + _epsillon, weightTwo, bias) - dloss)/_epsillon;
        var dlossTwo = (Loss(weightOne, weightTwo + _epsillon, bias) - dloss)/_epsillon;
        
        return (dlossOne, dlossTwo);
    }

    private double DBias(double weightOne, double weightTwo, double bias)
       => (Loss(weightOne, weightTwo, _epsillon + bias) 
        - Loss(weightOne, weightTwo, bias)) / _epsillon;

    private double Loss(double weightOne, double weightTwo, double bias)
    {
        double loss = 0.0;

        for (var index = 0; index < _trainingData.Length; index++)
        {
            var inputOne = _trainingData[index].A;
            var inputTwo = _trainingData[index].B;
            var output = MathUtils.Sigmoid((inputOne * weightOne) + (inputTwo * weightTwo) + bias);

            var actual = _trainingData[index].C;
            var errorMargin = output - actual;

            loss += errorMargin * errorMargin;
        }

        loss /= _trainingData.Length;

        return loss;
    }
}
