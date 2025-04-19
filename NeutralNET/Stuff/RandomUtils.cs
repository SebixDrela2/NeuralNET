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
}
