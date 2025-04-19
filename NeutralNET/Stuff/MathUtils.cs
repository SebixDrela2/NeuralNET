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
}
