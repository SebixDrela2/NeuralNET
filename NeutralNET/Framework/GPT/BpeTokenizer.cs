using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace NeutralNET.Framework.Neural.GPT;

/// <summary>
/// Byte-pair encoding tokenizer.
///
/// Pre-tokenization groups an optional single leading whitespace with the
/// following non-whitespace run, so tokens like " the" and " and" can form.
/// Merge application at encode time follows the standard "lowest-rank merge first"
/// rule, matching what was learned during training.
/// </summary>
public class BpeTokenizer
{
    private const string FileMagic = "BPE1";

    private readonly Dictionary<string, int> _tokenToId = new();
    private readonly Dictionary<int, string> _idToToken = new();
    private readonly Dictionary<(string, string), int> _mergeRank = new();

    public int VocabSize => _tokenToId.Count;

    // ------------------------------------------------------------------
    //  Training
    // ------------------------------------------------------------------

    public void Train(string text, int targetVocabSize)
    {
        _tokenToId.Clear();
        _idToToken.Clear();
        _mergeRank.Clear();

        // 1. Base vocabulary: every distinct character.
        int nextId = 0;
        foreach (char c in text.Distinct().OrderBy(c => c))
        {
            string s = c.ToString();
            _tokenToId[s] = nextId;
            _idToToken[nextId] = s;
            nextId++;
        }

        // 2. Unique words with frequencies.
        var freq = new Dictionary<string, int>();
        foreach (var w in SplitIntoWords(text))
            freq[w] = freq.GetValueOrDefault(w) + 1;

        // Convert each unique word into a mutable list of char tokens.
        var words = new List<(List<string> Tokens, int Freq)>(freq.Count);
        foreach (var kv in freq)
        {
            var chars = new List<string>(kv.Key.Length);
            foreach (char c in kv.Key) chars.Add(c.ToString());
            words.Add((chars, kv.Value));
        }

        // 3. Iterative merging.
        while (nextId < targetVocabSize)
        {
            // Count weighted pair frequencies.
            var pairCounts = new Dictionary<(string, string), long>();
            foreach (var (tokens, f) in words)
            {
                for (int i = 0; i < tokens.Count - 1; i++)
                {
                    var pair = (tokens[i], tokens[i + 1]);
                    pairCounts[pair] = pairCounts.GetValueOrDefault(pair) + f;
                }
            }
            if (pairCounts.Count == 0) break;

            // Pick the pair with the highest weighted count.
            var bestPair = default((string, string));
            long bestCount = 1; // require at least 2 occurrences
            foreach (var kv in pairCounts)
            {
                if (kv.Value > bestCount)
                {
                    bestCount = kv.Value;
                    bestPair = kv.Key;
                }
            }
            if (bestCount <= 1) break;

            string merged = bestPair.Item1 + bestPair.Item2;

            // Apply the merge in every word.
            for (int w = 0; w < words.Count; w++)
            {
                var (tokens, f) = words[w];
                var newTokens = new List<string>(tokens.Count);
                for (int i = 0; i < tokens.Count; i++)
                {
                    if (i < tokens.Count - 1
                        && tokens[i] == bestPair.Item1
                        && tokens[i + 1] == bestPair.Item2)
                    {
                        newTokens.Add(merged);
                        i++;
                    }
                    else
                    {
                        newTokens.Add(tokens[i]);
                    }
                }
                words[w] = (newTokens, f);
            }

            _tokenToId[merged] = nextId;
            _idToToken[nextId] = merged;
            _mergeRank[bestPair] = _mergeRank.Count;
            nextId++;
        }
    }

    /// <summary>
    /// Pre-tokenization.
    /// Yields, in order:
    ///  - a whole run of leading whitespace (minus one char) if longer than 1
    ///  - one whitespace char (if any) + the following non-whitespace run
    ///  - a trailing whitespace-only run at end of input
    /// </summary>
    private static IEnumerable<string> SplitIntoWords(string text)
    {
        int i = 0;
        while (i < text.Length)
        {
            int start = i;

            // Skip whitespace, remember where the body starts.
            int bodyStart = i;
            while (bodyStart < text.Length && char.IsWhiteSpace(text[bodyStart])) bodyStart++;

            if (bodyStart >= text.Length)
            {
                // Trailing whitespace.
                yield return text.Substring(start);
                yield break;
            }

            // Consume the non-whitespace body.
            i = bodyStart;
            while (i < text.Length && !char.IsWhiteSpace(text[i])) i++;

            int prefixLen = bodyStart - start;
            if (prefixLen > 1)
            {
                // Emit all but one whitespace char as its own token.
                yield return text.Substring(start, prefixLen - 1);
            }

            int wordStart = prefixLen > 0 ? bodyStart - 1 : start;
            yield return text.Substring(wordStart, i - wordStart);
        }
    }

    // ------------------------------------------------------------------
    //  Encoding
    // ------------------------------------------------------------------

    public List<int> Encode(string text)
    {
        var ids = new List<int>();
        if (string.IsNullOrEmpty(text)) return ids;

        foreach (var word in SplitIntoWords(text))
            EncodeWord(word, ids);
        return ids;
    }

    private void EncodeWord(string word, List<int> output)
    {
        // Start with one token per character.
        var tokens = new List<string>(word.Length);
        foreach (char c in word) tokens.Add(c.ToString());

        // Repeatedly apply the lowest-rank available merge.
        while (tokens.Count > 1)
        {
            int bestRank = int.MaxValue;
            int bestPos = -1;
            for (int i = 0; i < tokens.Count - 1; i++)
            {
                if (_mergeRank.TryGetValue((tokens[i], tokens[i + 1]), out int rank)
                    && rank < bestRank)
                {
                    bestRank = rank;
                    bestPos = i;
                }
            }
            if (bestPos < 0) break;

            tokens[bestPos] = tokens[bestPos] + tokens[bestPos + 1];
            tokens.RemoveAt(bestPos + 1);
        }

        // Look up final tokens.
        foreach (var t in tokens)
        {
            if (_tokenToId.TryGetValue(t, out int id))
            {
                output.Add(id);
            }
            else
            {
                // Fallback: per-char (shouldn't happen for in-vocab text).
                foreach (char c in t)
                    if (_tokenToId.TryGetValue(c.ToString(), out int cid))
                        output.Add(cid);
            }
        }
    }

    // ------------------------------------------------------------------
    //  Decoding
    // ------------------------------------------------------------------

    public string Decode(IEnumerable<int> tokens)
    {
        var sb = new StringBuilder();
        foreach (int t in tokens)
            if (_idToToken.TryGetValue(t, out var s))
                sb.Append(s);
        return sb.ToString();
    }

    // ------------------------------------------------------------------
    //  Persistence (binary, length-prefixed UTF-8)
    // ------------------------------------------------------------------

    public void Save(string path)
    {
        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        using var w = new BinaryWriter(File.Create(path));
        w.Write(FileMagic);
        w.Write(_tokenToId.Count);
        w.Write(_mergeRank.Count);

        foreach (var kv in _tokenToId.OrderBy(kv => kv.Value))
        {
            w.Write(kv.Value);
            WriteString(w, kv.Key);
        }

        foreach (var kv in _mergeRank.OrderBy(kv => kv.Value))
        {
            WriteString(w, kv.Key.Item1);
            WriteString(w, kv.Key.Item2);
        }
    }

    public void Load(string path)
    {
        _tokenToId.Clear();
        _idToToken.Clear();
        _mergeRank.Clear();

        using var r = new BinaryReader(File.OpenRead(path));
        string magic = r.ReadString();
        if (magic != FileMagic)
            throw new InvalidDataException($"Bad BPE file magic: '{magic}' (expected '{FileMagic}')");

        int vocabSize = r.ReadInt32();
        int mergeCount = r.ReadInt32();

        for (int i = 0; i < vocabSize; i++)
        {
            int id = r.ReadInt32();
            string token = ReadString(r);
            _tokenToId[token] = id;
            _idToToken[id] = token;
        }

        for (int i = 0; i < mergeCount; i++)
        {
            string a = ReadString(r);
            string b = ReadString(r);
            _mergeRank[(a, b)] = i;
        }
    }

    private static void WriteString(BinaryWriter w, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        w.Write(bytes.Length);
        w.Write(bytes);
    }

    private static string ReadString(BinaryReader r)
    {
        int len = r.ReadInt32();
        return Encoding.UTF8.GetString(r.ReadBytes(len));
    }
}
