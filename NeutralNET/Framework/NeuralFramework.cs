using NeutralNET.Matrices;
using NeutralNET.Models;

namespace NeutralNET.Framework;

public class NeuralFramework
{
    private ArraySegment<Matrix> MatrixInputs;
    private ArraySegment<Matrix> MatrixWeights;
    private ArraySegment<Matrix> MatrixBiases;

    private readonly int _count;
    private readonly int[] _architecture;

    public NeuralFramework(int[] architecture)
    {
        if (architecture.Length <= 0)
        {
            throw new ArgumentException("Negative or empty architecture.");
        }

        _architecture = architecture;      
        _count = architecture.Length - 1;

        Initialize();
    }

    private void Initialize()
    {
        MatrixInputs = new ArraySegment<Matrix>(new Matrix[_architecture.Length]);
        MatrixWeights = new ArraySegment<Matrix>(new Matrix[_count]);
        MatrixBiases = new ArraySegment<Matrix>(new Matrix[_count]);

        MatrixInputs[0] = new Matrix(1, _architecture[0]);

        for (var i = 1; i < _architecture.Length; i++)
        {
            MatrixWeights[i - 1] = new Matrix(MatrixInputs[i - 1].Columns, _architecture[i]);
            MatrixBiases[i - 1] = new Matrix(1, _architecture[i]);
            MatrixInputs[i] = new Matrix(1, _architecture[i]);
        }
    }
    
    public void Print(string name)
    {
        Console.WriteLine($"{name} = [");

        for (var i = 0; i < _count; i++)
        {
            MatrixWeights[i].Print($"{nameof(MatrixWeights)}[{i}]");
            MatrixBiases[i].Print($"{nameof(MatrixWeights)}[{i}]");
        }

        Console.WriteLine("]");
    }
    
    public void Run(NeuralFramework gradientFramework, IModel model)
    {
        float epsillon = 1e-1f;

        MatrixInputs[0].CopyDataFrom(model.TrainingInput.Row(1));

        Randomize();

        for (var i = 0; i < 1000 * 100; i++)
        {
            FiniteDifference(gradientFramework, epsillon, model.TrainingInput, model.TrainingOutput);
            Learn(gradientFramework);

            var loss = Loss(model.TrainingInput, model.TrainingOutput);
            Console.WriteLine($"Loss:{loss}");
        }

        Console.WriteLine();

        for (var i = 0; i < 2; i++)
        {
            for (var j = 0; j < 2; j++)
            {
                MatrixInputs[0].Set(0, 0, i);
                MatrixInputs[0].Set(0, 1, j);

                Forward();
                Console.WriteLine($"{i} ^ {j} = {MatrixInputs[_count].At(0, 0)}");
            }
        }
    }

    private void Learn(NeuralFramework gradient)
    {
        float rate = 1e-1f;

        for (var i = 0; i < _count; i++)
        {
            LearnInternal(MatrixWeights, gradient.MatrixWeights, rate, i);
            LearnInternal(MatrixBiases, gradient.MatrixBiases, rate, i);
        }
    }

    private void LearnInternal(
        in ArraySegment<Matrix> matrixes, 
        in ArraySegment<Matrix> gradientMatrixes,
        float rate,
        int index)
    {
        for (var j = 0; j < matrixes[index].Rows; j++)
        {
            for (var k = 0; k < matrixes[index].Columns; k++)
            {
                var computedRate = rate * gradientMatrixes[index].At(j, k);
                matrixes[index].Sub(j, k, computedRate);
            }
        }
    }

    private void FiniteDifference(
        NeuralFramework gradient, 
        float epsillon, 
        Matrix trainingInput, 
        Matrix trainingOutput)
    {
        var cost = Loss(trainingInput, trainingOutput);

        for (var i = 0; i < _count; i++)
        {
            ComputeGradient(MatrixWeights, gradient.MatrixWeights, trainingInput, trainingOutput, epsillon, cost, i);
            ComputeGradient(MatrixBiases, gradient.MatrixBiases, trainingInput, trainingOutput, epsillon, cost, i);           
        }
    }

    private void ComputeGradient(
        in ArraySegment<Matrix> matrixes, 
        in ArraySegment<Matrix> gradientMatrixes, 
        in Matrix trainingInput,
        in Matrix trainingOutput,
        float epsillon,
        float cost,
        int index)
    {
        for (var j = 0; j < matrixes[index].Rows; j++)
        {
            for (var k = 0; k < matrixes[index].Columns; k++)
            {
                var temp = matrixes[index].At(j, k);

                matrixes[index].Add(j, k, epsillon);

                var computedCost = (Loss(trainingInput, trainingOutput) - cost) / epsillon;
                gradientMatrixes[index].Set(j, k, computedCost);

                matrixes[index].Set(j, k, temp);
            }
        }
    }

    private float Loss(Matrix trainingInput, Matrix trainingOutput)
    {
        if (trainingInput.Rows != trainingOutput.Rows)
        {
            throw new NotImplementedException($"Training input rows: {trainingInput.Rows} is not training output rows: {trainingOutput.Rows}");
        }

        if (trainingOutput.Columns != MatrixInputs[_count].Columns)
        {
            throw new NotImplementedException($"Training output columns: {trainingInput.Rows} is not matrix output Columns: {MatrixInputs[_count].Columns}");
        }

        var cost = 0f;
        var outputColumns = trainingOutput.Columns;

        for (var i = 0; i < trainingInput.Rows; i++)
        {
            var inputRow = trainingInput.Row(i);
            var outputRow = trainingOutput.Row(i);

            MatrixInputs[0].CopyDataFrom(inputRow);
            Forward();

            for (var j = 0; j < outputColumns; j++)
            {
                float distance = MatrixInputs[_count].At(0, j) - outputRow.At(0, j);

                cost += distance * distance;
            }
        }

        return cost / trainingInput.Rows;
    }

    private void Forward()
    {
        for (var i = 0; i < _count; i++)
        {
            MatrixInputs[i + 1] = MatrixInputs[i].Dot(MatrixWeights[i]);
            MatrixInputs[i + 1].Sum(MatrixBiases[i]);
            MatrixInputs[i + 1].ApplySigmoid();
        }
    }

    private void Randomize(float low = 0, float high = 1)
    {
        for (var i = 0; i < _count; i++)
        {
            MatrixWeights[i].Randomize(low, high);
            MatrixBiases[i].Randomize(low, high);
        }
    }
}
