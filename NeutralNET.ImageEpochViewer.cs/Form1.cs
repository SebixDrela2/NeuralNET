using NeutralNET.Framework;
using NeutralNET.Framework.Neural;
using NeutralNET.Matrices;
using NeutralNET.Models;
using NeutralNET.Stuff;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace NeutralNET.ImageEpochViewer;

public partial class Form1 : Form
{
    private const int BatchSize = 1;
    private const int BitmapWidth = GraphicsUtils.Width;
    private const int BitmapHeight = GraphicsUtils.Height;

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
        _matrixes = _network.EnumerateEpochs().GetEnumerator();
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
        _model = new BitMapTransformationModel();
        _model.Prepare();

        _network = new NeuralNetworkBuilder<Architecture>()
            .WithArchitecture(
                inputSize: BitMapTransformationModel.PixelCount,
                hiddenLayers: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                outputSize: BitMapTransformationModel.PixelCount)
            .WithEpochs(2000)
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
        using var trueBitMap = new Bitmap(GraphicsUtils.Width, GraphicsUtils.Height, PixelFormat.Format32bppArgb);
        var index = 0;        

        for (int y = 0; y < BitmapHeight; y++)
        {
            for (int x = 0; x < BitmapWidth; x++, ++index)
            {
                var brightness = (byte)(values[index] * 0xFF);

                trueBitMap.SetPixel(x, y, Color.FromArgb(brightness, brightness, brightness));
            }
        }

        windowGraphics.InterpolationMode = InterpolationMode.NearestNeighbor;
        windowGraphics.SmoothingMode = SmoothingMode.HighQuality;

        windowGraphics.ScaleTransform(4, 4);
        windowGraphics.DrawImage(trueBitMap, 0, 0);
    }
}
