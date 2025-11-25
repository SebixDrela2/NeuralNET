public static partial class StringExtensions;

partial class StringExtensions
{
    public const char LineSeparator = '\n';

    extension(string)
    {
        public static string JoinLines<T>(ReadOnlySpan<string> values) => string.Join(LineSeparator, values);
        public static string JoinLines<T>(ReadOnlySpan<object?> values) => string.Join(LineSeparator, values);
        public static string JoinLines<T>(IEnumerable<T> values) => string.Join(LineSeparator, values);
        public static string JoinLines<T>(IEnumerable<T> values, Func<T, string> formater) => string.Join(LineSeparator, values.Select(formater));
    }

    public static string JoinToString<T>(this ReadOnlySpan<string?> values, char separator) => string.Join(separator, values);
    public static string JoinToString<T>(this ReadOnlySpan<string?> values, string? separator = null) => string.Join(separator, values);
    public static string JoinToString<T>(this ReadOnlySpan<object?> values, char separator) => string.Join(separator, values);
    public static string JoinToString<T>(this ReadOnlySpan<object?> values, string? separator = null) => string.Join(separator, values);
    public static string JoinToString<T>(this T values, char separator) where T : IEnumerable<string> => string.Join(separator, values);
    public static string JoinToString<T>(this T values, string? separator = null) where T : IEnumerable<string> => string.Join(separator, values);
    public static string JoinToString<T>(this IEnumerable<T> values, char separator) => string.Join(separator, values);
    public static string JoinToString<T>(this IEnumerable<T> values, string? separator = null) => string.Join(separator, values);
    public static string JoinToString<T>(this IEnumerable<T> values, Func<T, string> formater, char separator) => JoinToString(values.Select(formater), separator);
    public static string JoinToString<T>(this IEnumerable<T> values, Func<T, string> formater, string? separator = null) => JoinToString(values.Select(formater), separator);

    private static string DefaultFormater<T>(T? value) => value?.ToString() ?? "";
}
