using NeutralNET.Matrices;
using NeutralNET.Models;

namespace NeutralNET.Framework;

public class NeuralFramework
{
    private MatrixBatchProcessor _batchProcessor;

    private ArraySegment<Matrix> _matrixNeurons;
    private ArraySegment<Matrix> _matrixWeights;
    private ArraySegment<Matrix> _matrixBiases;

    private readonly NeuralNetworkConfig _config;

    public readonly int Count;

    public NeuralFramework(NeuralNetworkConfig config)
    {
        if (config.Architecture.Length <= 0)
        {
            throw new ArgumentException("Negative or empty architecture.");
        }

        _config = config;    
        Count = config.Architecture.Length - 1;

        Initialize();
    }

    private void Initialize()
    {
        _matrixNeurons = new ArraySegment<Matrix>(new Matrix[_config.Architecture.Length]);
        _matrixWeights = new ArraySegment<Matrix>(new Matrix[Count]);
        _matrixBiases = new ArraySegment<Matrix>(new Matrix[Count]);

        _matrixNeurons[0] = new Matrix(1, _config.Architecture[0]);

        for (var i = 1; i < _config.Architecture.Length; i++)
        {
            _matrixWeights[i - 1] = new Matrix(_matrixNeurons[i - 1].Columns, _config.Architecture[i]);
            _matrixBiases[i - 1] = new Matrix(1, _config.Architecture[i]);
            _matrixNeurons[i] = new Matrix(1, _config.Architecture[i]);
        }

        _batchProcessor = new MatrixBatchProcessor();
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
        var trainingInput = model.TrainingInput;
        var trainingOutput = model.TrainingOutput;

        _matrixNeurons[0].CopyDataFrom(trainingInput.Row(0));

        RandomizeWeights();
        HandleTraining(gradientFramework, trainingInput, trainingOutput);

        Console.WriteLine();

        return new ModelRunner
        {
            Input = _matrixNeurons[0],
            Forward = Forward
        };
    }

    private void HandleTraining(NeuralFramework gradientFramework, Matrix trainingInput, Matrix trainingOutput)
    {
        int[] indices = Enumerable.Range(0, trainingInput.Rows).ToArray();

        for (var epoch = 0; epoch < _config.Epochs; epoch++)
        {
            float loss = 0;

            ShuffleIndices(indices);

            var position = GetMatrixesPositioned(trainingInput, trainingOutput, indices);

            foreach (var (inputBatch, outputBatch) 
                in _batchProcessor.GetBatches(position.MatrixInput, position.MatrixOutput, _config.BatchSize))
            {
                ProcessBatch(gradientFramework, inputBatch, outputBatch, ref loss);
            }

            loss /= _config.BatchSize;

            if (epoch % 100 is 0)
            {
                Console.WriteLine($"Epoch ({epoch}/{_config.Epochs}) Loss: {loss}");
            }
        }
    }

    private (Matrix MatrixInput, Matrix MatrixOutput) GetMatrixesPositioned(Matrix input, Matrix output, int[] indices)
    {
        if (_config.WithShuffle)
        {
            return (input.Reorder(indices), output.Reorder(indices));
        }

        return (input, output);
    }

    private void ShuffleIndices(int[] indices)
    {
        var rand = new Random();

        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = rand.Next(i + 1);
            (indices[j], indices[i]) = (indices[i], indices[j]);
        }
    }

    private void ProcessBatch(
        NeuralFramework gradientFramework, 
        Matrix inputBatch, 
        Matrix outputBatch,
        ref float loss)
    {
        BackPropagate(gradientFramework, inputBatch, outputBatch);
        Learn(gradientFramework);

        loss += Loss(inputBatch, outputBatch);
    }

    private void Learn(NeuralFramework gradient)
    {
        for (var i = 0; i < Count; i++)
        {
            gradient._matrixWeights[i].Clip(-1f, 1f);
            gradient._matrixBiases[i].Clip(-1f, 1f);

            LearnInternal(_matrixWeights, gradient._matrixWeights, _config.LearningRate, i);
            LearnInternal(_matrixBiases, gradient._matrixBiases, _config.LearningRate, i);
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

                computedRate += rate * _config.WeightDecay * matrixes[index].At(j, k);
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

    [Obsolete]
    public void FiniteDifference(
        NeuralFramework gradient, 
        float epsillon, 
        Matrix trainingInput, 
        Matrix trainingOutput)
    {
        var loss = Loss(trainingInput, trainingOutput);

        for (var i = 0; i < Count; i++)
        {
            ComputeGradient(_matrixWeights, gradient._matrixWeights, trainingInput, trainingOutput, epsillon, loss, i);
            ComputeGradient(_matrixBiases, gradient._matrixBiases, trainingInput, trainingOutput, epsillon, loss, i);           
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
        float loss,
        int index)
    {
        for (var j = 0; j < matrixes[index].Rows; j++)
        {
            for (var k = 0; k < matrixes[index].Columns; k++)
            {
                var temp = matrixes[index].At(j, k);

                matrixes[index].Add(j, k, epsillon);

                var computedLoss = (Loss(trainingInput, trainingOutput) - loss) / epsillon;
                gradientMatrixes[index].Set(j, k, computedLoss);

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

        var loss = 0f;
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

                loss += distance * distance;
            }
        }

        return loss / trainingInput.Rows;
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

    private void RandomizeWeights()
    {
        for (var i = 0; i < Count; i++)
        {
            float scale = MathF.Sqrt(2f / _matrixWeights[i].Rows);
            _matrixWeights[i].Randomize(-scale, scale);
        }
    }
}
