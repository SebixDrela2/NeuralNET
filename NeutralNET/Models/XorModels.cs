using NeutralNET.Activation;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using System;

namespace NeutralNET.Models;

internal class XorModels
{
    public double X;
    public double Y;

    public double WeightOne = RandomUtils.GetDouble(1);
    public double WeightTwo = RandomUtils.GetDouble(1);
    public double BiasOne = RandomUtils.GetDouble(1);

    public double WeightThree = RandomUtils.GetDouble(1);
    public double WeightFour = RandomUtils.GetDouble(1);
    public double BiasTwo = RandomUtils.GetDouble(1);

    public double WeightFive = RandomUtils.GetDouble(1);
    public double WeightSix = RandomUtils.GetDouble(1);
    public double BiasThree = RandomUtils.GetDouble(1);
}

public unsafe class XorAdvanced
{
    public readonly float[] TrainingData = 
    [
        0, 0, 0,
        1, 0, 1,
        0, 1, 1,
        1, 1, 0,
    ];

    public NeuralMatrix TrainingInput { get; set; }
    public NeuralMatrix TrainingOutput { get; set; }

    public uint[] TrainingOutputStrideMask => throw new NotImplementedException();

    public NeuralMatrix A0 = new NeuralMatrix(1, 2);

    public NeuralMatrix W1 = new NeuralMatrix(2, 2);
    public NeuralMatrix B1 = new NeuralMatrix(1, 2);
    public NeuralMatrix W2 = new NeuralMatrix(2, 1);
    public NeuralMatrix B2 = new NeuralMatrix(1, 1);

    [Obsolete]
    public float Forward()
    {
        var A1 = A0.Dot(W1);
        A1.SumVectorized(B1);
        ActivationFunctions.ApplyReLUVectorized(A1);

        var A2 = A1.Dot(W2);
        A2.SumVectorized(B2);
        ActivationFunctions.ApplySigmoidVectorized(A2);

        return A2.Pointer[0];
    }

    public void Prepare()
    {
        TrainingInput = new NeuralMatrix(4, 2);

        TrainingData.AsSpan(0, 2).CopyTo(TrainingInput.GetRowSpan(0));
        TrainingData.AsSpan(3, 2).CopyTo(TrainingInput.GetRowSpan(1));
        TrainingData.AsSpan(6, 2).CopyTo(TrainingInput.GetRowSpan(2));
        TrainingData.AsSpan(9, 2).CopyTo(TrainingInput.GetRowSpan(3));

        TrainingOutput = new NeuralMatrix(4, 1);

        TrainingOutput.GetRowSpan(0)[0] = TrainingData[2];
        TrainingOutput.GetRowSpan(1)[0] = TrainingData[5];
        TrainingOutput.GetRowSpan(2)[0] = TrainingData[8];
        TrainingOutput.GetRowSpan(3)[0] = TrainingData[11];
    }

    public void Run(int x1, int x2)
    {
        float epsillon = 1e-4f;
        float rate = 1e-2f;

        W1.Randomize(-1f, 1f);
        B1.Randomize(-1f, 1f);
        W2.Randomize(-1f, 1f);
        B2.Randomize(-1f, 1f);

        A0.Set(0, 0, x1);
        A0.Set(0, 1, x2);

        TrainingInput.Print("input");
        TrainingOutput.Print("output");

        var gradient = new XorAdvanced();

        for (var i = 0; i < 200 * 1000; i++)
        {
            var loss = Loss(TrainingInput, TrainingOutput);
            Console.WriteLine($"Loss: {loss}");
            FiniteDifference(gradient, epsillon, TrainingInput, TrainingOutput);
            Learn(gradient, rate);
        }

        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                A0.Set(0, 0, i);
                A0.Set(0, 1, j);

                var value = Forward();
                Console.WriteLine($"{i} ^ {j} = {value}");
            }
        }
    }

    private void Learn(XorAdvanced gradient, float rate)
    {
        LearnInternal(W1, gradient.W1, rate);
        LearnInternal(B1, gradient.B1, rate);
        LearnInternal(W2, gradient.W2, rate);
        LearnInternal(B2, gradient.B2, rate);
    }

    private void LearnInternal(NeuralMatrix matrix, NeuralMatrix gradient, float rate)
    {
        for (var i = 0; i < matrix.Rows; i++)
        {
            for (var j = 0; j < matrix.UsedColumns; j++)
            {
                matrix.Sub(i, j, rate * gradient.At(i, j));
            }
        }
    }

    private void FiniteDifference(
        XorAdvanced gradient,
        float epsillon,
        NeuralMatrix trainingInput,
        NeuralMatrix trainingOutput)
    {
        float cost = Loss(trainingInput, trainingOutput);

        CalculateGradient(W1, gradient.W1, epsillon, cost, trainingInput, trainingOutput);
        CalculateGradient(B1, gradient.B1, epsillon, cost, trainingInput, trainingOutput);
        CalculateGradient(W2, gradient.W2, epsillon, cost, trainingInput, trainingOutput);
        CalculateGradient(B2, gradient.B2, epsillon, cost, trainingInput, trainingOutput);
    }

    private void CalculateGradient(
        NeuralMatrix matrix, 
        NeuralMatrix gradient, 
        float epsillon,
        float cost,
        NeuralMatrix trainingInput,
        NeuralMatrix trainingOutput)
    {
        for (var i = 0; i < matrix.Rows; i++)
        {
            for (var j = 0; j < matrix.UsedColumns; j++)
            {
                float originalValue = matrix.At(i, j);

                matrix.Set(i, j, originalValue + epsillon);
                float newCost = Loss(trainingInput, trainingOutput);

                float grad = (newCost - cost) / epsillon;
                gradient.Set(i, j, grad);

                matrix.Set(i, j, originalValue);
            }
        }
    }

    private float Loss(NeuralMatrix trainingInput, NeuralMatrix trainingOutput)
    {
        if (trainingInput.Rows != trainingOutput.Rows)
        {
            throw new NotImplementedException($"Training input rows: {trainingInput.Rows} is not training output rows: {trainingOutput.Rows}");
        }

        float cost = 0;

        for (var i = 0; i < trainingInput.Rows; i++)
        {
            var inputRow = trainingInput.GetRowSpan(i);
            var outputRow = trainingOutput.GetRowSpan(i);

            trainingInput.GetRowSpan(i).CopyTo(A0.GetRowSpan(0));
            var output = Forward();

            for (var j = 0; j < trainingOutput.UsedColumns; j++)
            {
                var distance = output - outputRow[j];

                cost += distance * distance;
            }
        }

        return cost / trainingInput.Rows;
    }
}
