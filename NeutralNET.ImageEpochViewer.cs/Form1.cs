using NeutralNET.Framework;
using NeutralNET.Framework.Neural;
using NeutralNET.Matrices;
using NeutralNET.Models;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace NeutralNET.ImageEpochViewer;

public partial class Form1 : Form
{
    private const int BatchSize = 1;
    private const int BitmapWidth = 16;
    private const int BitmapHeight = 16;

    private NeuralNetwork<Architecture> _network;
    private IModel _model;
    private IEnumerator<NeuralMatrix> _matrixes;
    private bool _isTraining;


    public float[]? BitMapValues => _isTraining ? _matrixes.Current.ToArray() : null;

    public Form1()
    {
        InitializeComponent();
    }

    public void Run()
    {
        Show();

        while (!IsDisposed)
        {
            Application.DoEvents();

            _isTraining = _matrixes.MoveNext();

            if (!_isTraining)
            {
                break;
            }

            Refresh();
            Thread.Sleep(5);
        }
    }

    protected override void OnLoad(EventArgs e)
    {
        Prepare();
        _matrixes = _network.RunEpoch().GetEnumerator();
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        base.OnPaint(e);

        if (BitMapValues is not null)
        {
            Draw(BitMapValues, e.Graphics);
        }
    }

    private void Prepare()
    {
        _model = new SingleDigitTransformationModel();
        _model.Prepare();

        _network = new NeuralNetworkBuilder<Architecture>()
            .WithArchitecture(
                inputSize: SingleDigitTransformationModel.PixelCount,
                hiddenLayers: [4, 4],
                outputSize: SingleDigitTransformationModel.PixelCount)
            .WithEpochs(10000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(1e-3f)
            .WithWeightDecay(3e-5f)
            .WithBeta1(0.9f)
            .WithBeta2(0.999f)
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .WithModel(_model)
            .Build();
    }

    private static void Draw(float[] values, Graphics windowGraphics)
    {
        using var trueBitMap = new Bitmap(16, 16, PixelFormat.Format32bppArgb);
        var index = 0;        

        for (int y = 0; y < BitmapHeight; y++)
        {
            for (int x = 0; x < BitmapWidth; x++, ++index)
            {
                var brightness = (byte)(values[index] * 255);

                trueBitMap.SetPixel(x, y, Color.FromArgb(brightness, brightness, brightness));
            }
        }

        windowGraphics.InterpolationMode = InterpolationMode.NearestNeighbor;
        windowGraphics.SmoothingMode = SmoothingMode.HighQuality;

        windowGraphics.ScaleTransform(16, 16);
        windowGraphics.DrawImage(trueBitMap, 0, 0);

    }
}
