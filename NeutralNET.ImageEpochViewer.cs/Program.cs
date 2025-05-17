namespace NeutralNET.ImageEpochViewer;

internal static class Program
{
    [STAThread]
    static void Main()
    {
        ApplicationConfiguration.Initialize();
        var mainForm = new Form1();

        mainForm.Run();
    }
}