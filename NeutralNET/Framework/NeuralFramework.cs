using NeutralNET.Matrices;
using NeutralNET.Models;
using NeutralNET.Utils;
using System.Diagnostics;

namespace NeutralNET.Framework;

public class NeuralFramework
{
    private MatrixBatchProcessor _batchProcessor = null!;

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

        _matrixNeurons[0].CopyRowFrom(trainingInput, 0);

        RandomizeWeights();

        var start = TimerUtils.TimeStamp;

        HandleTraining(gradientFramework, trainingInput, trainingOutput);

        var stop = TimeSpan.FromSeconds((TimerUtils.TimeStamp - start) / (double)Stopwatch.Frequency);

        Console.WriteLine($"Time: {stop.TotalMilliseconds}");

        Console.WriteLine();

        return new ModelRunner
        {
            Input = _matrixNeurons[0],
            Forward = Forward
        };
    }

    private void HandleTraining(NeuralFramework gradientFramework, Matrix trainingInput, Matrix trainingOutput)
    {
        var rng = new Random();
        int[] indices = [.. Enumerable.Range(0, trainingInput.Rows)];

        for (var epoch = 0; epoch < _config.Epochs; epoch++)
        {
            float loss = 0;

            if (_config.WithShuffle)
            {
                rng.Shuffle(indices);
            }

            var rows = indices.Select(i => (
                Input: trainingInput.GetRowMemory(i),
                Output: trainingOutput.GetRowMemory(i)
            ));

            var batches = _batchProcessor.GetBatches(
                rows,
                trainingInput.Rows,
                _config.BatchSize
            );

            foreach (var batch in batches)
            {
                ProcessBatch(gradientFramework, batch, ref loss);
            }

            loss /= _config.BatchSize;

            if (epoch % _config.BatchSize is 0)
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

    // private void ShuffleIndices(Random rngSource, int[] indices)
    // {
    //     //rngSource.Shuffle()
    //     for (int i = indices.Length - 1; i > 0; i--)
    //     {
    //         int j = rngSource.Next(i + 1);
    //         (indices[j], indices[i]) = (indices[i], indices[j]);
    //     }
    // }

    private void ProcessBatch(
        NeuralFramework gradientFramework, 
        IEnumerable<(Memory<float> Input, Memory<float> Output)> batch,
        ref float loss)
    {
        BackPropagate(gradientFramework, batch);
        Learn(gradientFramework);

        loss += Loss(batch);
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
        IEnumerable<(Memory<float> Input, Memory<float> Output)> batch)
    {       
        gradient.ZeroOut();

        int count = 0;
        foreach (var pair in batch)
        {
            var inputRow = pair.Input.Span;
            var outputRow = pair.Output.Span;

            inputRow.CopyTo(_matrixNeurons[0].Span);
            Forward();

            for (var j = 0; j < Count ; j++)
            {
                gradient._matrixNeurons[j].Fill(0);
            }

            ComputeOutputLayer(gradient, outputRow);
            PropagateToPreviousLayer(gradient);

            ++count;
        }

        NormalizeGradients(gradient, count);
    }

    private void ComputeOutputLayer(NeuralFramework gradient, Span<float> outputRow)
    {
        for(int index = 0; index < outputRow.Length; ++index)
        {
            var difference = _matrixNeurons[Count].Span[index] - outputRow[index];
            gradient._matrixNeurons[Count].Span[index] = difference;
        }
    }

    private void PropagateToPreviousLayer(NeuralFramework gradient)
    {       
        for (int layerIdx = Count; layerIdx > 0; layerIdx--)
        {            
            var currentActivations = _matrixNeurons[layerIdx].GetRowSpan(0);
            var currentErrors = gradient._matrixNeurons[layerIdx].GetRowSpan(0);

            BackPropagateLayer(layerIdx, gradient, currentActivations, currentErrors);
        }
    }

    private void BackPropagateLayer(
        int layerIndex,
        NeuralFramework gradient,
        ReadOnlySpan<float> currentActivations,
        ReadOnlySpan<float> currentErrors)
    {
        bool isOutputLayer = layerIndex == Count;

        for (int neuronIdx = 0; neuronIdx < currentActivations.Length; neuronIdx++)
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

    private void NormalizeGradients(
        NeuralFramework gradient,
        int rowN)
    {
        for (var i = 0; i < gradient.Count; i++)
        {
            for (var j = 0; j < gradient._matrixWeights[i].Rows; j++)
            {
                for (var k = 0; k < gradient._matrixWeights[i].Columns; k++)
                {
                    gradient._matrixWeights[i].Divide(j, k, rowN);
                }
            }

            for (var j = 0; j < gradient._matrixBiases[i].Rows; j++)
            {
                for (var k = 0; k < gradient._matrixBiases[i].Columns; k++)
                {
                    gradient._matrixBiases[i].Divide(j, k, rowN);
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
    private void FiniteDifference(
        NeuralFramework gradient, 
        float epsillon, 
        Matrix trainingInput, 
        Matrix trainingOutput)
    {
        throw new NotSupportedException();
        // var loss = Loss(trainingInput, trainingOutput);

        // for (var i = 0; i < Count; i++)
        // {
        //     ComputeGradient(_matrixWeights, gradient._matrixWeights, trainingInput, trainingOutput, epsillon, loss, i);
        //     ComputeGradient(_matrixBiases, gradient._matrixBiases, trainingInput, trainingOutput, epsillon, loss, i);           
        // }
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
        throw new NotSupportedException();
        // for (var j = 0; j < matrixes[index].Rows; j++)
        // {
        //     for (var k = 0; k < matrixes[index].Columns; k++)
        //     {
        //         var temp = matrixes[index].At(j, k);

        //         matrixes[index].Add(j, k, epsillon);

        //         var computedLoss = (Loss(trainingInput, trainingOutput) - loss) / epsillon;
        //         gradientMatrixes[index].Set(j, k, computedLoss);

        //         matrixes[index].Set(j, k, temp);
        //     }
        // }
    }

    private float Loss(IEnumerable<(Memory<float> Input, Memory<float> Output)> batch)
    {
        var loss = 0f;       
        var count = 0;

        foreach (var pair in batch)
        {
            var inputRow = pair.Input.Span;
            var outputRow = pair.Output.Span;
            var outputColumns = outputRow.Length;
            
            inputRow.CopyTo(_matrixNeurons[0].Span);

            Forward();

            for (int j = 0; j < outputColumns; ++j)
            {
                float distance = _matrixNeurons[Count].Span[j] - outputRow[j];
                loss += distance * distance;
            }

            ++count;
        }

        return loss / count;
    }

    private Matrix Forward()
    {
        for (var i = 0; i < Count; i++)
        {
            _matrixNeurons[i].Dot(_matrixWeights[i], _matrixNeurons[i + 1]);
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
