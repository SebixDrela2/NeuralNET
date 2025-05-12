using System.Drawing;

namespace NeutralNET.Utils;

public static class ConsoleExtensions
{
    public static string WithColor(this string value, Color color) => $"\e[38;2;{color.R};{color.G};{color.B}m{value}\e[0m";
}
