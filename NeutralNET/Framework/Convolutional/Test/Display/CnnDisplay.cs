using NeutralNET.Framework.Convolutional;
using NeutralNET.Matrices;
using NeutralNET.Test.Data;

namespace NeutralTest;

public class CnnDisplayWriter(char[] items, int dataSetSize)
{
    private const int LabelColSize = 3;
    private const int LabelPadL = (LabelColSize - 1) / 2;
    private const int LabelPadR = LabelColSize / 2;

    public char[] Items { get; } = items;
    private readonly string _txtBorderMid = new string('═', items.Length * LabelColSize);
    private readonly string _txtLabels = string.Join("", items.Select(c => $"{"",LabelPadL}{c}{"",LabelPadR}"));

    public int Epoch { get; set; }
    public float Accuracy
    {
        get;
        set
        {
            field = value;
            if (field <= BestAccuracy) return;
            EpochsSinceBest = 0;
            BestAccuracy = field;
        }
    }
    public float AvgLoss => TotalLoss / dataSetSize;
    public float TotalLoss { get; set; }
    public float BestAccuracy { get; set; }
    public int EpochsSinceBest { get; set; }

    public void Clear()
    {
        Console.Write("\e[2J\e[3J\e[H");
    }

    public void Update(ReadOnlySpan<float> xss)
    {
        Epoch += 1;

        const string Sep1 = "══════════════╤══════════════════╤════════════════════╤═════════════════";
        const string Sep2 = "══════════════╧══════════════════╧════════════════════╧═════════════════";

        Console.Write("\e[H");

        Console.WriteLine($"╔{Sep1}╗\e[K");
        Console.WriteLine($"║  Epoch {Epoch,5} │ Loss: {AvgLoss,9:F6}  │  Accuracy: {Accuracy,7:P2} │  Best: {BestAccuracy,7:P2}  ║\e[K");
        Console.WriteLine($"╠{Sep2}╝\e[K");
        Console.WriteLine($"║{_txtLabels}│\e[A\e[D╤\e[B\e[K");
        Console.WriteLine($"╠{_txtBorderMid}╡\e[K");

        while (xss is [var x, .. var xs])
        {
            var probs = xs[..Items.Length];
            xss = xs[Items.Length..];
            var predicted = ArgMax(probs);
            var actual = (int)x;

            char predChar = (char)('A' + predicted);
            char actualChar = actual >= 0 ? (char)('A' + actual) : '?';

            Console.Write($"║");
            for (int j = 0; j < Items.Length; j++)
            {
                Console.Write($"{FmtPogression(probs[j], j == actual)}");
            }

            bool isOk = predicted == actual;
            var mark = isOk ? AsGreen("✓") : AsRed("✗");
            var lhs = predChar.ToString();
            var rhs = actualChar.ToString();

            Console.WriteLine($"│ {lhs} {mark} {rhs}\e[K");

        }

        Console.WriteLine($"╚{_txtBorderMid}╛\e[K");
        Console.WriteLine();
        Console.WriteLine($"Best accuracy: {BestAccuracy:P2}  |  Epochs since best: {EpochsSinceBest}\e[K");

        EpochsSinceBest += 1;
    }

    private static class Chars
    {
        public const char Zero = ' ';
        public const char Max = '\u2588';

        public const char Good = '✓';
        public const char Bad = '✗';
    }
    private static string FmtPogression(float x, bool hl)
    {
        const string bg = "12;12;12";
        const string fg = "163;163;163";
        const string errBg = "227;61;48";
        const string errFg = "151;41;32";

        const string hlBg = "10;12;13";
        const string hlFg = "78;91;106";
        const string hlErrBg = "98;67;75";
        const string hlErrFg = "227;61;48";

        const string bgBad = "35;11;0";
        const string bgGood = "0;35;11";
        const string fgBad = "163;53;0";
        const string fgGood = "0;163;53";

        switch (hl, x)
        {
            case (true, 0): return $"\e[38;2;{fgBad}m{new string(Chars.Bad, LabelColSize)}\e[39m";
            case (true, 1): return $"\e[38;2;0;53;163m{new string(Chars.Max, LabelColSize)}\e[39m";

            case (false, 0): return $"\e[48;2;{bg}m{new string(Chars.Zero, LabelColSize)}\e[49m";
            case (false, 1): return $"\e[38;2;{fg}m{new string(Chars.Max, LabelColSize)}\e[39m";

            case (false, <= 0): return $"\e[38;2;{errBg}m{x,5:f2}\e[39m";
            case (false, >= 1): return $"\e[38;2;{errFg};48;2;{errBg}m{x,5:f3}\e[39;49m";

            case (true, <= 0): return $"\e[38;2;{hlErrBg}m{x,5:f2}\e[39m";
            case (true, >= 1): return $"\e[38;2;{hlErrFg};48;2;{hlErrBg}m{x,5:f3}\e[39;49m";
        }// ✓ ✗
        Span<char> xs = stackalloc char[LabelColSize];
        xs.Fill(Chars.Zero);

        if (x >= 1e-12f)
        {
            var scaled = x * LabelColSize;
            var maxEnd = int.Clamp((int)scaled, 0, LabelColSize);
            xs[..maxEnd].Fill(Chars.Max);

            if (maxEnd != LabelColSize)
            {
                int frame = int.Clamp((int)((8 * (scaled - maxEnd)) + 0.5f), 0, 7);
                xs[maxEnd] = (char)(Chars.Max + (7 - frame));
            }
        }

        if (!hl) return $"\e[48;2;35;35;35;38;2;{fg}m{xs}\e[39;49m";

        switch (x)
        {
            case < 0.1f: return $"\e[48;2;35;11;0;38;2;163;53;0m{xs}\e[39;49m";
            case > 0.9f: return $"\e[48;2;0;35;11;38;2;0;163;53m{xs}\e[39;49m";
        }

        return $"\e[48;2;168;152;46;38;2;245;222;67m{xs}\e[39;49m";
    }

    private static int ArgMax(ReadOnlySpan<float> array)
    {
        int maxIdx = 0;
        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > array[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }

    private static string AsGreenOrRed(bool cnd, string x) => cnd ? AsGreen(x) : AsRed(x);
    private static string AsGreen(string x) => $"\e[38;2;124;179;66m{x}\e[39m";
    private static string AsRed(string x) => $"\e[38;2;230;74;25m{x}\e[39m";
}

