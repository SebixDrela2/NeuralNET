namespace NeutralNET.Framework.Neural.CNN;

public class SamplePrediction
{
    public int SampleIndex { get; set; }
    public int Predicted { get; set; }
    public int Actual { get; set; }
    public bool IsCorrect { get; set; }
    public float[] Probabilities { get; set; } = Array.Empty<float>();
}
