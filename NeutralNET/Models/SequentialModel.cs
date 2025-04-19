using NeutralNET.Stuff;

namespace NeutralNET.Models;

public class SequentialModel
{
    private readonly (int A, int B)[] _trainingData;    

    private readonly double _rate;
    private readonly double _epsillon;
    private readonly int _trainingLength;

    public SequentialModel(       
        (int A, int B)[] trainingData,
        double rate, 
        double epsillion,
        int trainingLength,
        double bias)
    {       
        _trainingData = trainingData;
        _rate = rate;
        _epsillon = epsillion;
        _trainingLength = trainingLength;
    }

    public void Run()
    {
        var weight = RandomUtils.GetDouble(10);
        var bias = RandomUtils.GetDouble(5);

        for (var index = 0; index < _trainingLength; index++) 
        {
            var dloss = DLoss(weight, bias);
            var dBias = DBias(weight, bias);
            weight -= _rate * dloss;

            var loss = DLoss(weight, bias);
            Console.WriteLine($"Current loss: {loss}");
        }

        Console.WriteLine(weight);
    }

    private double DLoss(double weight, double bias) 
        => (Loss(weight + _epsillon, bias) - Loss(weight, bias)) / _epsillon;

    private double DBias(double weight, double bias)
       => (Loss(weight ,_epsillon + bias) - Loss(weight, bias)) / _epsillon;

    private double Loss(double weight, double bias)
    {
        double loss = 0.0;

        for (var index = 0; index < _trainingData.Length; index++)
        {
            var input = _trainingData[index].A;
            var output = input * weight;

            var actual = _trainingData[index].B;
            var errorMargin = output - actual;

            loss += errorMargin * errorMargin;
        }

        loss /= _trainingData.Length;

        return loss;
    }
}
