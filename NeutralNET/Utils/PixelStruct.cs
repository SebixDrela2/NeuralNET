namespace NeutralNET.Utils;

public struct PixelStruct(int digit, int size)
{
    private int Digit = digit;

    public float[] Values = new float[size];
    public readonly float MappedValue => (Digit / 9f);
}
