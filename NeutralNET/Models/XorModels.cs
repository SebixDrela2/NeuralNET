using NeutralNET.Matrices;
using NeutralNET.Stuff;

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

internal class XorAdvanced
{
    public readonly ArraySegment<float> TrainingData = new(
    [
        0, 0, 0,
        1, 0, 1,
        0, 1, 1,
        1, 1, 0,
    ]);

    public Matrix TrainingInput;
    public Matrix TrainingOutput;

    public Matrix A0 = new Matrix(1, 2);
    public Matrix W1 = new Matrix(2, 2);
    public Matrix B1 = new Matrix(1, 2);
    public Matrix W2 = new Matrix(2, 1);
    public Matrix B2 = new Matrix(1, 1);

    public float Forward()
    {
        var A1 = A0.Dot(ref W1);
        A1.Sum(ref B1);
        A1.ApplySigmoid();

        var A2 = A1.Dot(ref W2);
        A2.Sum(ref B2);
        A2.ApplySigmoid();

        return A2.FirstElement;
    }

    public void Prepare()
    {
        var trainingData = new Matrix(4, 3)
        {
            Data = TrainingData
        };

        TrainingInput = trainingData.SplitStart(2);
        TrainingOutput = trainingData.SplitEnd(3);
    }

    public void Run(int x1, int x2)
    {
        float epsillon = 1e-1f;
        float rate = 1e-0f;

        W1.Randomize();
        B1.Randomize();
        W2.Randomize();
        B2.Randomize();

        A0.Set(0, 0, x1);
        A0.Set(0, 1, x2);

        TrainingInput.Print("input");
        TrainingOutput.Print("output");


        var gradient = new XorAdvanced();

        for (var i = 0; i < 2000 * 1000; i++)
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

    private void LearnInternal(Matrix matrix, Matrix gradient, float rate)
    {
        for (var i = 0; i < matrix.Rows; i++)
        {
            for (var j = 0; j < matrix.Columns; j++)
            {
                matrix.Sub(i, j, rate * gradient.At(i, j));
            }
        }
    }

    private void FiniteDifference(
        XorAdvanced gradient,
        float epsillon,
        Matrix trainingInput,
        Matrix trainingOutput)
    {
        float cost = Loss(trainingInput, trainingOutput);

        CalculateGradient(W1, gradient.W1, epsillon, cost, trainingInput, trainingOutput);
        CalculateGradient(B1, gradient.B1, epsillon, cost, trainingInput, trainingOutput);
        CalculateGradient(W2, gradient.W2, epsillon, cost, trainingInput, trainingOutput);
        CalculateGradient(B2, gradient.B2, epsillon, cost, trainingInput, trainingOutput);
    }

    private void CalculateGradient(
        Matrix matrix, 
        Matrix gradient, 
        float epsillon,
        float cost,
        Matrix trainingInput,
        Matrix trainingOutput)
    {
        for (var i = 0; i < matrix.Rows; i++)
        {
            for (var j = 0; j < matrix.Columns; j++)
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

    private float Loss(Matrix trainingInput, Matrix trainingOutput)
    {
        if (trainingInput.Rows != trainingOutput.Rows)
        {
            throw new NotImplementedException($"Training input rows: {trainingInput.Rows} is not training output rows: {trainingOutput.Rows}");
        }

        float cost = 0;

        for (var i = 0; i < trainingInput.Rows; i++)
        {
            var inputRow = trainingInput.Row(i);
            var outputRow = trainingOutput.Row(i);

            A0.CopyDataFrom(inputRow);
            var output = Forward();

            for (var j = 0; j < trainingOutput.Columns; j++)
            {
                var distance = output - outputRow.At(0, j);

                cost += distance * distance;
            }
        }

        return cost / trainingInput.Rows;
    }
}
