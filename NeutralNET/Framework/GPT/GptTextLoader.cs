namespace NeutralNET.Framework.Neural.GPT;

/// <summary>
/// Loads a local text corpus, builds a character vocabulary mapping,
/// and extracts overlapping sliding-window sequence batches for autoregressive model training.
/// </summary>
public class GPTTextDataLoader
{
    private readonly Dictionary<char, int> _charToId = new();
    private readonly Dictionary<int, char> _idToChar = new();

    public int VocabSize => _charToId.Count;
    public List<int> EncodedTokens { get; private set; } = new();

    public void LoadCorpus(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"[GPTTextDataLoader] File not found: {filePath}");
        }

        var rawText = File.ReadAllText(filePath);
        var slicedText = rawText.Substring(0, (int)(rawText.Length * 0.3));

        if (!slicedText.Contains("To be, or not to be"))
        {
            throw new InvalidOperationException($"No shakespearing.");
        }

        if (string.IsNullOrWhiteSpace(slicedText))
        {
            throw new InvalidOperationException($"[GPTTextDataLoader] Corpus file '{filePath}' is empty.");
        }

        BuildVocabulary(slicedText);
        EncodedTokens = Encode(slicedText);

        Console.WriteLine($"[GPTTextDataLoader] Loaded file: {filePath}");
        Console.WriteLine($"[GPTTextDataLoader] Total characters: {slicedText.Length:N0} | Vocab size: {VocabSize}");
    }

    private void BuildVocabulary(string text)
    {
        var uniqueChars = text.Distinct().OrderBy(c => c).ToList();
        _charToId.Clear();
        _idToChar.Clear();

        for (int i = 0; i < uniqueChars.Count; i++)
        {
            _charToId[uniqueChars[i]] = i;
            _idToChar[i] = uniqueChars[i];
        }
    }

    public List<int> Encode(string text)
    {
        List<int> tokens = new(text.Length);
        foreach (char c in text)
        {
            if (_charToId.TryGetValue(c, out int id))
            {
                tokens.Add(id);
            }
        }
        return tokens;
    }

    public string Decode(IEnumerable<int> tokens)
    {
        return new string(tokens.Select(id => _idToChar.TryGetValue(id, out char c) ? c : '?').ToArray());
    }

    public List<int[]> GetBatches(int contextSize, int stride = -1)
    {
        if (stride <= 0) stride = contextSize / 2;

        List<int[]> batches = new();
        for (int i = 0; i < EncodedTokens.Count - contextSize; i += stride)
        {
            int[] sequence = new int[contextSize + 1];
            EncodedTokens.CopyTo(i, sequence, 0, contextSize + 1);
            batches.Add(sequence);
        }

        return batches;
    }
}
