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
    private Bitmap _backBuffer;
    private Graphics _backGraphics;


    public float[]? BitMapValues => _isTraining ? _matrixes.Current.ToArray() : null;

    public Form1()
    {
        InitializeComponent();

        ClientSize = new Size(BitmapWidth, BitmapHeight);
        StartPosition = FormStartPosition.CenterScreen;

        DoubleBuffered = false;
        SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint, true);
        UpdateStyles();

        _backBuffer = new Bitmap(BitmapWidth, BitmapHeight, PixelFormat.Format32bppArgb);
        _backGraphics = Graphics.FromImage(_backBuffer);
        _backGraphics.InterpolationMode = InterpolationMode.NearestNeighbor;
        _backGraphics.SmoothingMode = SmoothingMode.None;
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
        if (BitMapValues is not null)
        {
            Draw(BitMapValues);
            base.OnPaint(e);
            e.Graphics.DrawImageUnscaled(_backBuffer, 0, 0);
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

    private void Draw(float[] values)
    {
        var index = 0;

        BitmapData data = _backBuffer.LockBits(
            new Rectangle(0, 0, BitmapWidth, BitmapHeight),
            ImageLockMode.WriteOnly,
            PixelFormat.Format32bppArgb);

        unsafe
        {
            byte* ptr = (byte*)data.Scan0;
            for (int y = 0; y < BitmapHeight; y++)
            {
                for (int x = 0; x < BitmapWidth; x++, index++)
                {
                    byte brightness = (byte)(values[index] * 255);

                    int offset = (y * data.Stride) + (x * 4);
                    ptr[offset + 0] = brightness; // B
                    ptr[offset + 1] = brightness; // G
                    ptr[offset + 2] = brightness; // R
                    ptr[offset + 3] = 255;        // A
                }
            }
        }

        _backBuffer.UnlockBits(data);
    }
}
