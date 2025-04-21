using NeutralNET.Matrices;
using NeutralNET.Models;

namespace NeutralNET.Framework;

public class NeuralFramework
{
    public readonly int Count;

    private const int TrainingCount = 100 * 200;

    private const float Rate = 1e-2f;
    private const float WeightDecay = 1e-4f;

    private ArraySegment<Matrix> _matrixNeurons;
    private ArraySegment<Matrix> _matrixWeights;
    private ArraySegment<Matrix> _matrixBiases;
    
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
        _matrixNeurons = new ArraySegment<Matrix>(new Matrix[_architecture.Length]);
        _matrixWeights = new ArraySegment<Matrix>(new Matrix[Count]);
        _matrixBiases = new ArraySegment<Matrix>(new Matrix[Count]);

        _matrixNeurons[0] = new Matrix(1, _architecture[0]);

        for (var i = 1; i < _architecture.Length; i++)
        {
            _matrixWeights[i - 1] = new Matrix(_matrixNeurons[i - 1].Columns, _architecture[i]);
            _matrixBiases[i - 1] = new Matrix(1, _architecture[i]);
            _matrixNeurons[i] = new Matrix(1, _architecture[i]);
        }
    }
    
    public void Print(string name)
    {
        Console.WriteLine($"{name} = [");

        for (var i = 0; i < Count; i++)
        {
            _matrixWeights[i].Print($"{nameof(_matrixWeights)}[{i}]");
            _matrixBiases[i].Print($"{nameof(_matrixWeights)}[{i}]");
        }

        Console.WriteLine("]");
    }
    
    public IModelRunner Run(NeuralFramework gradientFramework, IModel model)
    {
        _matrixNeurons[0].CopyDataFrom(model.TrainingInput.Row(1));

        Randomize();

        for (var index = 0; index < TrainingCount; index++)
        {
            BackPropagate(gradientFramework, model.TrainingInput, model.TrainingOutput);
            Learn(gradientFramework);

            if (index % 100 is 0)
            {
                var loss = Loss(model.TrainingInput, model.TrainingOutput);
                Console.WriteLine($"Loss {index + 1}:{loss}");
            }
        }

        Console.WriteLine();

        return new ModelRunner
        {
            Input = _matrixNeurons[0],
            Forward = Forward
        };
    }

    private void Learn(NeuralFramework gradient)
    {
        for (var i = 0; i < Count; i++)
        {
            gradient._matrixWeights[i].Clip(-1f, 1f);
            gradient._matrixBiases[i].Clip(-1f, 1f);

            LearnInternal(_matrixWeights, gradient._matrixWeights, Rate, i);
            LearnInternal(_matrixBiases, gradient._matrixBiases, Rate, i);
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

                computedRate += rate * WeightDecay * matrixes[index].At(j, k);
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

        if (_matrixNeurons[Count].Columns != trainingOutput.Columns)
        {
            throw new NotImplementedException($"Output columns: {_matrixNeurons[Count].Columns} is not training output columns: {trainingOutput.Columns}");
        }

        gradient.ZeroOut();

        for (var index = 0; index < trainingInput.Rows; index++)
        {
            _matrixNeurons[0].CopyDataFrom(trainingInput.Row(index));
            Forward();
            
            for (var j = 0; j < Count ; j++)
            {
                gradient._matrixNeurons[j].Fill(0);
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
            var difference = _matrixNeurons[Count].At(0, j) - trainingOutput.At(index, j);
            gradient._matrixNeurons[Count].Set(0, j, difference);
        }
    }

    private void PropagateToPreviousLayer(NeuralFramework gradient)
    {       
        for (int layerIdx = Count; layerIdx > 0; layerIdx--)
        {            
            var currentActivations = _matrixNeurons[layerIdx].Row(0).Data;
            var currentErrors = gradient._matrixNeurons[layerIdx].Row(0).Data;

            BackPropagateLayer(layerIdx, gradient, currentActivations, currentErrors);
        }
    }

    private void BackPropagateLayer(
    int layerIndex,
    NeuralFramework gradient,
    ArraySegment<float> currentActivations,
    ArraySegment<float> currentErrors)
    {
        bool isOutputLayer = layerIndex == Count;

        for (int neuronIdx = 0; neuronIdx < currentActivations.Count; neuronIdx++)
        {
            float activation = currentActivations[neuronIdx];
            float error = currentErrors[neuronIdx];

            var neuronGradient = CalculateNeuronGradient(activation, error, useRelu: !isOutputLayer);

            gradient._matrixBiases[layerIndex - 1].Add(0, neuronIdx, neuronGradient);

            for (int prevNeuronIdx = 0; prevNeuronIdx < _matrixNeurons[layerIndex - 1].Columns; prevNeuronIdx++)
            {
                float prevActivation = _matrixNeurons[layerIndex - 1].At(0, prevNeuronIdx);
                gradient._matrixWeights[layerIndex - 1].Add(
                    prevNeuronIdx,
                    neuronIdx,
                    neuronGradient * prevActivation);

                gradient._matrixNeurons[layerIndex - 1].Add(
                    0,
                    prevNeuronIdx,
                    neuronGradient * _matrixWeights[layerIndex - 1].At(prevNeuronIdx, neuronIdx));
            }
        }
    }

    private void NormalizeGradients(NeuralFramework gradient, in Matrix trainingInput)
    {
        for (var i = 0; i < gradient.Count; i++)
        {
            for (var j = 0; j < gradient._matrixWeights[i].Rows; j++)
            {
                for (var k = 0; k < gradient._matrixWeights[i].Columns; k++)
                {
                    gradient._matrixWeights[i].Divide(j, k, trainingInput.Rows);
                }
            }

            for (var j = 0; j < gradient._matrixBiases[i].Rows; j++)
            {
                for (var k = 0; k < gradient._matrixBiases[i].Columns; k++)
                {
                    gradient._matrixBiases[i].Divide(j, k, trainingInput.Rows);
                }
            }
        }
    }

    private float CalculateNeuronGradient(float activation, float error, bool useRelu)
    {
        if (useRelu)
        {            
            return 2 * error * (activation > 0 ? 1 : 0);
        }
        
        return 2 * error;
    }

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
            ComputeGradient(_matrixWeights, gradient._matrixWeights, trainingInput, trainingOutput, epsillon, cost, i);
            ComputeGradient(_matrixBiases, gradient._matrixBiases, trainingInput, trainingOutput, epsillon, cost, i);           
        }
    }

    private void ZeroOut()
    {
        for (var i = 0; i < Count; i++)
        {
            _matrixNeurons[i].Fill(0);
            _matrixWeights[i].Fill(0);
            _matrixBiases[i].Fill(0);
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

        if (trainingOutput.Columns != _matrixNeurons[Count].Columns)
        {
            throw new NotImplementedException($"Training output columns: {trainingInput.Rows} is not matrix output Columns: {_matrixNeurons[Count].Columns}");
        }

        var cost = 0f;
        var outputColumns = trainingOutput.Columns;

        for (var i = 0; i < trainingInput.Rows; i++)
        {
            var inputRow = trainingInput.Row(i);
            var outputRow = trainingOutput.Row(i);

            _matrixNeurons[0].CopyDataFrom(inputRow);
            
            Forward();

            for (var j = 0; j < outputColumns; j++)
            {
                float distance = _matrixNeurons[Count].At(0, j) - outputRow.At(0, j);

                cost += distance * distance;
            }
        }

        return cost / trainingInput.Rows;
    }

    private Matrix Forward()
    {
        for (var i = 0; i < Count; i++)
        {
            _matrixNeurons[i + 1] = _matrixNeurons[i].Dot(_matrixWeights[i]);
            _matrixNeurons[i + 1].Sum(_matrixBiases[i]);

            if (i < Count - 1)
            {
                _matrixNeurons[i + 1].ApplyReLU();
            }
            else
            {
                _matrixNeurons[i + 1].ApplySigmoid();
            }
        }
        return _matrixNeurons[Count];
    }

    private void Randomize()
    {
        for (var i = 0; i < Count; i++)
        {
            float scale = MathF.Sqrt(2f / _matrixWeights[i].Rows);
            _matrixWeights[i].Randomize(-scale, scale);
        }
    }
}
