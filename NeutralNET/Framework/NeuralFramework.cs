using NeutralNET.Matrices;
using NeutralNET.Models;

namespace NeutralNET.Framework;

public class NeuralFramework
{
    public readonly int Count;

    private ArraySegment<Matrix> MatrixInputs;
    private ArraySegment<Matrix> MatrixWeights;
    private ArraySegment<Matrix> MatrixBiases;
    
    private readonly int[] _architecture;

    public NeuralFramework(int[] architecture)
    {
        if (architecture.Length <= 0)
        {
            throw new ArgumentException("Negative or empty architecture.");
        }

        _architecture = architecture;      
        Count = architecture.Length - 1;

        Initialize();
    }

    private void Initialize()
    {
        MatrixInputs = new ArraySegment<Matrix>(new Matrix[_architecture.Length]);
        MatrixWeights = new ArraySegment<Matrix>(new Matrix[Count]);
        MatrixBiases = new ArraySegment<Matrix>(new Matrix[Count]);

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

        for (var i = 0; i < Count; i++)
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
            BackPropagate(gradientFramework, model.TrainingInput, model.TrainingOutput);
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
                Console.WriteLine($"{i} ^ {j} = {MatrixInputs[Count].At(0, 0)}");
            }
        }
    }

    private void Learn(NeuralFramework gradient)
    {
        float rate = 1e-1f;

        for (var i = 0; i < Count; i++)
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

    private void BackPropagate(
        NeuralFramework gradient,
        Matrix trainingInput,
        Matrix trainingOutput)
    {
        if (trainingInput.Rows != trainingOutput.Rows)
        {
            throw new NotImplementedException($"Training input rows: {trainingInput.Rows} is not training output rows: {trainingOutput.Rows}");
        }

        if (MatrixInputs[Count].Columns != trainingOutput.Columns)
        {
            throw new NotImplementedException($"Output columns: {MatrixInputs[Count].Columns} is not training output columns: {trainingOutput.Columns}");
        }

        for (var i = 0; i < trainingInput.Rows; i++)
        {
            MatrixInputs[0].CopyDataFrom(trainingInput.Row(i));
            Forward();
            
            for (var j = 0; j <  trainingOutput.Columns; j++)
            {
                var difference = MatrixInputs[Count].At(0, j) - trainingOutput.At(i, j);
                gradient.MatrixInputs[Count].Set(0, j, difference);
            }

            for (var l = Count; l > 0; l--)
            {
                for (var j = 0; j < MatrixInputs[l].Columns; j++)
                {
                    float a = MatrixInputs[l].At(0, j);
                    float differenceA = gradient.MatrixInputs[l].At(0, j);

                    var computed = 2 * differenceA * a * (1 - a);
                    gradient.MatrixBiases[l - 1].Add(0, j, computed);

                    for (var k = 0; k < MatrixInputs[l - 1].Columns; k++)
                    {
                        float previousLayer = MatrixInputs[l - 1].At(0, k);
                        var weight = MatrixWeights[l - 1].At(k, j);

                        var computedLayer = computed * previousLayer;
                        var computedWeight = computed * weight;

                        gradient.MatrixWeights[l - 1].Set(k, j, computedLayer);
                        gradient.MatrixInputs[l - 1].Add(0, k, computedWeight);
                    }
                }
            }
        }

        for (var i = 0; i < gradient.Count; i++)
        {
            for (var j = 0; j < gradient.MatrixWeights[i].Rows; j++)
            {
                for (var k = 0; k < gradient.MatrixWeights[i].Columns; k++)
                {
                    gradient.MatrixWeights[i].Divide(j, k, trainingInput.Rows);
                }
            }

            for (var j = 0; j < gradient.MatrixBiases[i].Rows; j++)
            {
                for (var k = 0; k < gradient.MatrixBiases[i].Columns; k++)
                {
                    gradient.MatrixBiases[i].Divide(j, k, trainingInput.Rows);
                }
            }
        }
    }

    [Obsolete]
    public void FiniteDifference(
        NeuralFramework gradient, 
        float epsillon, 
        Matrix trainingInput, 
        Matrix trainingOutput)
    {
        var cost = Loss(trainingInput, trainingOutput);

        for (var i = 0; i < Count; i++)
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

        if (trainingOutput.Columns != MatrixInputs[Count].Columns)
        {
            throw new NotImplementedException($"Training output columns: {trainingInput.Rows} is not matrix output Columns: {MatrixInputs[Count].Columns}");
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
                float distance = MatrixInputs[Count].At(0, j) - outputRow.At(0, j);

                cost += distance * distance;
            }
        }

        return cost / trainingInput.Rows;
    }

    private void Forward()
    {
        for (var i = 0; i < Count; i++)
        {
            MatrixInputs[i + 1] = MatrixInputs[i].Dot(MatrixWeights[i]);
            MatrixInputs[i + 1].Sum(MatrixBiases[i]);
            MatrixInputs[i + 1].ApplySigmoid();
        }
    }

    private void Randomize(float low = 0, float high = 1)
    {
        for (var i = 0; i < Count; i++)
        {
            MatrixWeights[i].Randomize(low, high);
            MatrixBiases[i].Randomize(low, high);
        }
    }
}
