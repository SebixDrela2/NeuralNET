using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeutralNET.Framework.Neural.GPT;

public class GPTTextDataLoader
{
    // Char-level state
    private readonly Dictionary<char, int> _charToId = new();
    private readonly Dictionary<int, char> _idToChar = new();

    // BPE state
    private BpeTokenizer? _bpe;

    /// <summary>If true, use BPE; otherwise char-level.</summary>
    public bool UseBpe { get; set; } = true;

    /// <summary>Target vocab size when training a fresh BPE model.</summary>
    public int BpeVocabSize { get; set; } = 512;

    /// <summary>Optional explicit path for the BPE vocab file.</summary>
    public string? BpeVocabPath { get; set; }

    public int VocabSize => UseBpe ? (_bpe?.VocabSize ?? 0) : _charToId.Count;
    public List<int> EncodedTokens { get; private set; } = new();

    public void LoadCorpus(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"[GPTTextDataLoader] File not found: {filePath}");

        var rawText = File.ReadAllText(filePath);
        var slicedText = rawText; // full corpus (change back to *0.3 if you want)

        if (string.IsNullOrWhiteSpace(slicedText))
            throw new InvalidOperationException($"[GPTTextDataLoader] Corpus file '{filePath}' is empty.");

        if (UseBpe)
        {
            LoadBpe(slicedText, filePath);
        }
        else
        {
            BuildVocabulary(slicedText);
            EncodedTokens = Encode(slicedText);
        }

        Console.WriteLine($"[GPTTextDataLoader] Loaded file: {filePath}");
        Console.WriteLine($"[GPTTextDataLoader] Total characters: {slicedText.Length:N0} | Vocab size: {VocabSize}");
    }

    private void LoadBpe(string text, string corpusPath)
    {
        string vocabPath = BpeVocabPath ?? (corpusPath + ".bpe");
        _bpe = new BpeTokenizer();

        if (File.Exists(vocabPath))
        {
            _bpe.Load(vocabPath);
            Console.WriteLine($"[BPE] Loaded vocab from {vocabPath} ({_bpe.VocabSize} tokens)");
        }
        else
        {
            Console.WriteLine($"[BPE] Training vocab on {text.Length:N0} chars, target {BpeVocabSize}...");
            _bpe.Train(text, BpeVocabSize);
            _bpe.Save(vocabPath);
            Console.WriteLine($"[BPE] Trained vocab: {_bpe.VocabSize} tokens → {vocabPath}");
        }

        EncodedTokens = _bpe.Encode(text);
        Console.WriteLine($"[BPE] Encoded corpus to {EncodedTokens.Count:N0} tokens " +
                          $"(avg {text.Length / (double)EncodedTokens.Count:F2} chars/token)");
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
        if (UseBpe && _bpe != null) return _bpe.Encode(text);

        List<int> tokens = new(text.Length);
        foreach (char c in text)
            if (_charToId.TryGetValue(c, out int id))
                tokens.Add(id);
        return tokens;
    }

    public string Decode(IEnumerable<int> tokens)
    {
        if (UseBpe && _bpe != null) return _bpe.Decode(tokens);
        return new string(tokens.Select(id => _idToChar.TryGetValue(id, out char c) ? c : '?').ToArray());
    }

    public List<int[]> GetBatches(int contextSize, int stride = -1)
    {
        if (stride <= 0) stride = contextSize / 2;

        List<int[]> batches = new();
        for (int i = 0; i < EncodedTokens.Count - contextSize; i += stride)
        {
            int[] seq = new int[contextSize + 1];
            EncodedTokens.CopyTo(i, seq, 0, contextSize + 1);
            batches.Add(seq);
        }
        return batches;
    }
}
