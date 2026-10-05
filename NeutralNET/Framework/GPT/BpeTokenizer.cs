using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace NeutralNET.Framework.Neural.GPT;

/// <summary>
/// A lightweight Byte-Pair Encoding (BPE) tokenizer supporting vocabulary training, 
/// saving/loading merges, and encoding/decoding raw string inputs.
/// </summary>
public class BpeTokenizer
{
    private readonly Dictionary<string, int> _encoder = [];
    private readonly Dictionary<int, string> _decoder = [];
    private readonly List<(string, string)> _merges = [];
    private int _vocabSize;

    public int VocabSize => _encoder.Count;

    public BpeTokenizer() { }

    /// <summary>
    /// Trains BPE vocabulary directly on a text corpus up to targetVocabSize.
    /// </summary>
    public void Train(string text, int targetVocabSize)
    {
        _encoder.Clear();
        _decoder.Clear();
        _merges.Clear();

        // 1. Initialize base vocabulary with character-level tokens
        HashSet<char> uniqueChars = new(text);
        int currentId = 0;
        foreach (char c in uniqueChars)
        {
            string token = c.ToString();
            if (!_encoder.ContainsKey(token))
            {
                _encoder[token] = currentId;
                _decoder[currentId] = token;
                currentId++;
            }
        }

        // Represent initial text as words split into character sequence lists
        List<List<string>> words = new();
        string[] rawTokens = text.Split(new[] { ' ', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var word in rawTokens)
        {
            List<string> chars = new();
            foreach (char c in word)
            {
                chars.Add(c.ToString());
            }
            if (chars.Count > 0) words.Add(chars);
        }

        // 2. Iteratively merge most frequent adjacent pairs
        while (_encoder.Count < targetVocabSize)
        {
            Dictionary<(string, string), int> pairCounts = new();

            foreach (var word in words)
            {
                for (int i = 0; i < word.Count - 1; i++)
                {
                    var pair = (word[i], word[i + 1]);
                    pairCounts[pair] = pairCounts.GetValueOrDefault(pair, 0) + 1;
                }
            }

            if (pairCounts.Count == 0) break;

            // Find pair with max frequency
            (string, string) bestPair = ("", "");
            int maxFreq = -1;
            foreach (var kvp in pairCounts)
            {
                if (kvp.Value > maxFreq)
                {
                    maxFreq = kvp.Value;
                    bestPair = kvp.Key;
                }
            }

            if (maxFreq <= 1) break;

            string newToken = bestPair.Item1 + bestPair.Item2;
            _encoder[newToken] = currentId;
            _decoder[currentId] = newToken;
            _merges.Add(bestPair);
            currentId++;

            // Replace occurrences of bestPair in text
            for (int w = 0; w < words.Count; w++)
            {
                var word = words[w];
                List<string> newWord = new();
                for (int i = 0; i < word.Count; i++)
                {
                    if (i < word.Count - 1 && word[i] == bestPair.Item1 && word[i + 1] == bestPair.Item2)
                    {
                        newWord.Add(newToken);
                        i++; // Skip merged element
                    }
                    else
                    {
                        newWord.Add(word[i]);
                    }
                }
                words[w] = newWord;
            }
        }

        _vocabSize = _encoder.Count;
    }

    public List<int> Encode(string text)
    {
        List<int> tokens = new();
        if (string.IsNullOrEmpty(text)) return tokens;

        int idx = 0;
        while (idx < text.Length)
        {
            int longestMatchLength = 0;
            int matchedId = -1;

            // Greedy match longest token present in encoder
            foreach (var kvp in _encoder)
            {
                if (kvp.Key.Length > longestMatchLength && text.AsSpan(idx).StartsWith(kvp.Key))
                {
                    longestMatchLength = kvp.Key.Length;
                    matchedId = kvp.Value;
                }
            }

            if (matchedId != -1)
            {
                tokens.Add(matchedId);
                idx += longestMatchLength;
            }
            else
            {
                // Fallback unknown token handling via character key addition
                string unkChar = text[idx].ToString();
                if (!_encoder.TryGetValue(unkChar, out int id))
                {
                    id = _encoder.Count;
                    _encoder[unkChar] = id;
                    _decoder[id] = unkChar;
                }
                tokens.Add(id);
                idx++;
            }
        }

        return tokens;
    }

    public string Decode(List<int> tokens)
    {
        StringBuilder sb = new();
        foreach (var t in tokens)
        {
            if (_decoder.TryGetValue(t, out string? val))
            {
                sb.Append(val);
            }
        }
        return sb.ToString();
    }

    public void SaveVocabulary(string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine(_encoder.Count);
        foreach (var kvp in _encoder)
        {
            writer.WriteLine($"{kvp.Value}\t{kvp.Key}");
        }
    }

    public void LoadVocabulary(string path)
    {
        _encoder.Clear();
        _decoder.Clear();
        using var reader = new StreamReader(path);
        int count = int.Parse(reader.ReadLine() ?? "0");
        for (int i = 0; i < count; i++)
        {
            var line = reader.ReadLine()?.Split('\t');
            if (line != null && line.Length >= 2)
            {
                int id = int.Parse(line[0]);
                string token = line[1];
                _encoder[token] = id;
                _decoder[id] = token;
            }
        }
    }
}
