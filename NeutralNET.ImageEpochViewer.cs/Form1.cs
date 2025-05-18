using NeutralNET.Activation;
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
    private const int BatchSize = 32;
    private const int BitmapWidth = GraphicsUtils.Width;
    private const int BitmapHeight = GraphicsUtils.Height;
    private const int ScaleFactor = 256/GraphicsUtils.Width;
    private NeuralNetwork<Architecture> _network;
    private IModel _model;
    private IEnumerator<NeuralMatrix> _matrixes;
    private Bitmap _backBuffer;
    private Graphics _backGraphics;
    private readonly System.Windows.Forms.Timer _refreshTimer = new();


    public float[]? BitMapValues => GetBitMapValues();

    public Form1()
    {
        InitializeComponent();

        ClientSize = new Size(256, 256);
        StartPosition = FormStartPosition.CenterScreen;

        DoubleBuffered = true;
        SetStyle(ControlStyles.OptimizedDoubleBuffer |
                 ControlStyles.AllPaintingInWmPaint |
                 ControlStyles.UserPaint, true);
        UpdateStyles();

        _backBuffer = new Bitmap(256, 256, PixelFormat.Format32bppArgb);
        _backGraphics = Graphics.FromImage(_backBuffer);
        _backGraphics.InterpolationMode = InterpolationMode.NearestNeighbor;
        _backGraphics.SmoothingMode = SmoothingMode.None;

        _refreshTimer.Interval = 16;
        _refreshTimer.Tick += (s, e) => Refresh();
    }

    public void Run()
    {
        Show();
        _refreshTimer.Start();

        while (!IsDisposed)
        {
            Application.DoEvents();

            Thread.Sleep(500);
        }
    }

    protected override async void OnLoad(EventArgs e)
    {
        Prepare();
       await Task.Run(_network.Run);
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        if (BitMapValues != null && _backBuffer != null)
        {
            DrawGrayscale(BitMapValues);
            e.Graphics.DrawImageUnscaled(_backBuffer, 0, 0);
        }
    }

    protected override void OnPaintBackground(PaintEventArgs e)
    {
        // Do nothing to prevent background flicker
    }

    private void Prepare()
    {
        _model = new GrayScaleImageModel();
        _model.Prepare();

        _network = new NeuralNetworkBuilder<Architecture>(_model)
            .WithArchitecture([64, 32, 32, 16])
            .WithEpochs(1000000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(3e-4f)
            .WithHiddenLayerActivation(ActivationType.LeakyReLU)
            .WithOutputLayerActivation(ActivationType.ReLU)
            .WithWeightDecay(1e-5f)
            .WithBeta1(0.9f)
            .WithBeta2(0.999f)
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .Build();
    }

    private float[] GetBitMapValues()
    {
        var outputs = new float[BitmapWidth * BitmapHeight];
        float normX = 1f / (BitmapWidth - 1);
        float normY = 1f / (BitmapHeight - 1);

        for (int x = 0; x < BitmapWidth; x++)
        {
            for (int y = 0; y < BitmapHeight; y++)
            {
                float[] input = [x * normX, y * normY];
                var inputSpan = _network.Architecture.MatrixNeurons[0].GetRowSpan(0);
                input.CopyTo(inputSpan);
                
                outputs[x * BitmapHeight + y] = _network.Forward().GetRowSpan(0)[0];
            }
        }
        return outputs;
    }

    private unsafe void DrawGrayscale(float[] values)
    {
        var rect = new Rectangle(0, 0, 256, 256);
        var data = _backBuffer.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format32bppPArgb);

        for (int origY = 0; origY < BitmapHeight; origY++)
        {
            for (int origX = 0; origX < BitmapWidth; origX++)
            {
                int origIdx = origY * BitmapWidth + origX;
                byte brightness = (byte)(values[origIdx] * 255);

                for (int yOffset = 0; yOffset < ScaleFactor; yOffset++)
                {
                    int displayY = origY * ScaleFactor + yOffset;
                    byte* row = (byte*)data.Scan0 + displayY * data.Stride;

                    for (int xOffset = 0; xOffset < ScaleFactor; xOffset++)
                    {
                        int displayX = origX * ScaleFactor + xOffset;
                        int pixelPos = displayX * 4;

                        row[pixelPos + 0] = brightness;
                        row[pixelPos + 1] = brightness;
                        row[pixelPos + 2] = brightness;
                        row[pixelPos + 3] = 255;       
                    }
                }
            }
        }

        _backBuffer.UnlockBits(data);
    }

    public unsafe void DrawRGB(float[] values)
    {
        var rect = new Rectangle(0, 0, BitmapWidth, BitmapHeight);
        var data = _backBuffer.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format32bppPArgb);

        Parallel.For(0, BitmapHeight, y =>
        {
            byte* row = (byte*)data.Scan0 + y * data.Stride;
            for (int x = 0; x < BitmapWidth; x++)
            {
                int rgbIdx = (y * BitmapWidth + x) * 3;

                row[x * 4 + 0] = (byte)(values[rgbIdx + 2] * 255);
                row[x * 4 + 1] = (byte)(values[rgbIdx + 1] * 255);
                row[x * 4 + 2] = (byte)(values[rgbIdx] * 255);
                row[x * 4 + 3] = 255;
            }
        });

        _backBuffer.UnlockBits(data);
    }
}
