namespace NeutralNET.Matrices;

public class BatchNormLayer
{
    public NeuralMatrix Gamma;
    public NeuralMatrix Beta;
    public NeuralMatrix RunningMean;
    public NeuralMatrix RunningVar;

    public float Momentum = 0.1f;
    public float Epsilon = 1e-5f;

    public BatchNormLayer(int size)
    {
        Gamma = new NeuralMatrix(1, size);
        Beta = new NeuralMatrix(1, size);
        RunningMean = new NeuralMatrix(1, size);
        RunningVar = new NeuralMatrix(1, size);

        Gamma.Fill(1.0f);
        Beta.Fill(0.0f);
    }
}
