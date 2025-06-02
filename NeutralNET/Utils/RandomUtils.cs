namespace NeutralNET.Stuff;

public static class RandomUtils
{
    private static readonly Random _random = new();

    public static double GetDouble(double multiplier, int? seed = null)
    {
        if (seed is null)
        {
            return _random.NextDouble() * multiplier;
        }

        return new Random((int)seed).NextDouble() * multiplier;
    }

    public static float GetFloat(double multiplier, int? seed = null)
    {
        return (float) GetDouble(multiplier, seed);
    }

    public static float GetGaussian(float mean = 0f, float stddev = 1f, float multiplier = 1f, int? seed = null)
    {
        float u1 = 1.0f - GetFloat(multiplier, seed);
        float u2 = GetFloat(multiplier, seed);

        float randStdNormal = MathF.Sqrt(-2.0f * MathF.Log(u1)) *
                              MathF.Sin(2.0f * MathF.PI * u2);

        return mean + stddev * randStdNormal;
    }
}
