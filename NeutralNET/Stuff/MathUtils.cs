namespace NeutralNET.Stuff;

public static class MathUtils
{
    public static double Sigmoid(double value)
    {
        return 1 / (1 + Math.Exp(-value));
    }
    public static float Sigmoid(float value)
    {
        return (float)(1 / (1 + Math.Exp(-value)));
    }

    public static double ReLU(double value)
    {
        return Math.Max(0, value);
    }

    public static float ReLU(float value)
    {
        return Math.Max(0, value);
    }
}
