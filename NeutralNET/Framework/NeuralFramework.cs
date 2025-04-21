using NeutralNET.Matrices;
using NeutralNET.Models;

namespace NeutralNET.Framework;

public class NeuralFramework
{
    public readonly int Count;

    private ArraySegment<Matrix> MatrixNeurons;
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
        MatrixNeurons = new ArraySegment<Matrix>(new Matrix[_architecture.Length]);
        MatrixWeights = new ArraySegment<Matrix>(new Matrix[Count]);
        MatrixBiases = new ArraySegment<Matrix>(new Matrix[Count]);

        MatrixNeurons[0] = new Matrix(1, _architecture[0]);

        for (var i = 1; i < _architecture.Length; i++)
        {
            MatrixWeights[i - 1] = new Matrix(MatrixNeurons[i - 1].Columns, _architecture[i]);
            MatrixBiases[i - 1] = new Matrix(1, _architecture[i]);
            MatrixNeurons[i] = new Matrix(1, _architecture[i]);
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
        MatrixNeurons[0].CopyDataFrom(model.TrainingInput.Row(1));

        Randomize();

        for (var i = 0; i < 200 * 100; i++)
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
                MatrixNeurons[0].Set(0, 0, i);
                MatrixNeurons[0].Set(0, 1, j);

                Forward();
                Console.WriteLine($"{i} ^ {j} = {MatrixNeurons[Count].At(0, 0)}");
            }
        }
    }

    private void Learn(NeuralFramework gradient)
    {
        float rate = 1;

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

        if (MatrixNeurons[Count].Columns != trainingOutput.Columns)
        {
            throw new NotImplementedException($"Output columns: {MatrixNeurons[Count].Columns} is not training output columns: {trainingOutput.Columns}");
        }

        gradient.ZeroOut();

        for (var index = 0; index < trainingInput.Rows; index++)
        {
            MatrixNeurons[0].CopyDataFrom(trainingInput.Row(index));
            Forward();
            
            for (var j = 0; j < Count ; j++)
            {
                gradient.MatrixNeurons[j].Fill(0);
            }

            ComputeOutputLayer(gradient, trainingOutput, index);
            PropagateToPreviousLayer(gradient);
        }

        NormalizeGradients(gradient, trainingInput);
    }

    private void ComputeOutputLayer(NeuralFramework gradient, in Matrix trainingOutput, int index)
    {
        for (var j = 0; j < trainingOutput.Columns; j++)
        {
            var difference = MatrixNeurons[Count].At(0, j) - trainingOutput.At(index, j);
            gradient.MatrixNeurons[Count].Set(0, j, difference);
        }
    }

    private void PropagateToPreviousLayer(NeuralFramework gradient)
    {       
        for (int layerIdx = Count; layerIdx > 0; layerIdx--)
        {            
            var currentActivations = MatrixNeurons[layerIdx].Row(0).Data;
            var currentErrors = gradient.MatrixNeurons[layerIdx].Row(0).Data;

            BackpropagateLayer(layerIdx, gradient, currentActivations, currentErrors);
        }
    }

    private void BackpropagateLayer(
    int layerIndex,
    NeuralFramework gradient,
    ArraySegment<float> currentActivations,
    ArraySegment<float> currentErrors)
    {
        for (int neuronIdx = 0; neuronIdx < currentActivations.Count; neuronIdx++)
        {
            float neuronGradient = CalculateNeuronGradient(
                currentActivations[neuronIdx],
                currentErrors[neuronIdx]);

            gradient.MatrixBiases[layerIndex - 1].Add(0, neuronIdx, neuronGradient);

            for (int prevNeuronIdx = 0; prevNeuronIdx < MatrixNeurons[layerIndex - 1].Columns; prevNeuronIdx++)
            {
                float prevActivation = MatrixNeurons[layerIndex - 1].At(0, prevNeuronIdx);
                float weight = MatrixWeights[layerIndex - 1].At(prevNeuronIdx, neuronIdx);

                gradient.MatrixWeights[layerIndex - 1].Add(
                    prevNeuronIdx,
                    neuronIdx,
                    CalculateWeightGradient(neuronGradient, prevActivation));
                
                gradient.MatrixNeurons[layerIndex - 1].Add(
                    0,
                    prevNeuronIdx,
                    CalculatePreviousLayerError(neuronGradient, weight));
            }
        }
    }

    private void NormalizeGradients(NeuralFramework gradient, in Matrix trainingInput)
    {
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

    private float CalculateNeuronGradient(float activation, float error) => 2 * error * activation * (1 - activation);
    private float CalculateWeightGradient(float neuronGradient, float previousActivation) => neuronGradient * previousActivation;
    private float CalculatePreviousLayerError(float neuronGradient, float connectionWeight) => neuronGradient * connectionWeight;


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

    private void ZeroOut()
    {
        for (var i = 0; i < Count; i++)
        {
            MatrixNeurons[i].Fill(0);
            MatrixWeights[i].Fill(0);
            MatrixBiases[i].Fill(0);
        }
    }

    [Obsolete]
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

        if (trainingOutput.Columns != MatrixNeurons[Count].Columns)
        {
            throw new NotImplementedException($"Training output columns: {trainingInput.Rows} is not matrix output Columns: {MatrixNeurons[Count].Columns}");
        }

        var cost = 0f;
        var outputColumns = trainingOutput.Columns;

        for (var i = 0; i < trainingInput.Rows; i++)
        {
            var inputRow = trainingInput.Row(i);
            var outputRow = trainingOutput.Row(i);

            MatrixNeurons[0].CopyDataFrom(inputRow);
            Forward();

            for (var j = 0; j < outputColumns; j++)
            {
                float distance = MatrixNeurons[Count].At(0, j) - outputRow.At(0, j);

                cost += distance * distance;
            }
        }

        return cost / trainingInput.Rows;
    }

    private void Forward()
    {
        for (var i = 0; i < Count; i++)
        {
            MatrixNeurons[i + 1] = MatrixNeurons[i].Dot(MatrixWeights[i]);
            MatrixNeurons[i + 1].Sum(MatrixBiases[i]);
            MatrixNeurons[i + 1].ApplySigmoid();
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
