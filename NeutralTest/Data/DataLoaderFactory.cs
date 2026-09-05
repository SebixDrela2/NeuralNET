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
            DataSourceType.Cifar10 => new Cifar10DataLoader(),
            DataSourceType.DigiDigi => new DigiDigiDataLoader(),
            _ => throw new ArgumentException($"Unsupported data source type: {sourceType}")
        };
    }
}
