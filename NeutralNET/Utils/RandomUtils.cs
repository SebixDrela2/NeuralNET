namespace NeutralNET.Stuff;

public static class RandomUtils
{
    // private static readonly Random _random = Random.Shared;
    private static readonly Random _rng = new Random(0x6547);

    public static double GetDouble(double multiplier = 1, int? seed = null)
    {
        // return .5;
        if (seed is null)
        {
            lock (_rng)
            {
                return _rng.NextDouble() * multiplier;
            }
        }

        throw new NotImplementedException(); // DEBUG later
        //return new Random((int)seed).NextDouble() * multiplier;
    }

    public static float GetFloat(double multiplier = 1, int? seed = null)
    {
        // return .5f;
        return (float)GetDouble(multiplier, seed);
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
