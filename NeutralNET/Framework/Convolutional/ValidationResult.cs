namespace NeutralNET.Framework.Neural.CNN;

public class ValidationResult
{
    public float Accuracy { get; set; }
    public int Correct { get; set; }
    public int Total { get; set; }
    public List<SamplePrediction> SamplePredictions { get; set; } = new();
}
