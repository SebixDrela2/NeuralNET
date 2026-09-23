namespace NeutralNET.Test.Data;


public static class DataLoaderFactory
{
    /// <summary>
    /// Creates a data loader for the specified data source type
    /// </summary>
    public static DataLoaderBase Create(DataSourceType sourceType)
    {
        return sourceType switch
        {
            // DataSourceType.Cifar10 => new Cifar10DataLoader(),
            // DataSourceType.DigiDigi => new DigiDigiDataLoader(),
            // DataSourceType.MNIST => new MNISTDataLoader(),
            DataSourceType.Letters => OperatingSystem.IsWindowsVersionAtLeast(6, 1) ? new LetterDataLoader() : throw new NotSupportedException(),
            _ => throw new ArgumentException($"Unsupported data source type: {sourceType}")
        };
    }

    /// <summary>
    /// Creates a data loader for the specified data source type
    /// </summary>
    public static T Create<T>()
        where T : DataLoaderBase, IDataLoader<T>
    {
        return T.Create();
    }
}
