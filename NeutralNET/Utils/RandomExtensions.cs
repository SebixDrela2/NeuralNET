namespace NeutralNET.Utils;

public static class RandomExtensions
{
    public static double NextGaussian(this Random random, double mean = 0, double stddev = 1)
    {
        // Box-Muller transform
        double u1 = 1.0 - random.NextDouble(); // avoid zero
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stddev * randStdNormal;
    }

    public static float NextGaussianFloat(this Random random, float mean = 0, float stddev = 1)
    {
        return (float)random.NextGaussian(mean, stddev);
    }
}
