using NeutralNET.Activation;
using NeutralNET.Framework;
using NeutralNET.Framework.Neural;
using NeutralNET.Matrices;
using NeutralNET.Models;
using NeutralNET.Stuff;
using System.Diagnostics;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace NeutralNET.ImageEpochViewer;

public partial class Form1 : Form
{
    private const int BatchSize = 32;
    private const int BitmapWidth = GraphicsUtils.Width;
    private const int BitmapHeight = GraphicsUtils.Height;
    private const int ScaleFactor = 256 / GraphicsUtils.Width;
    private const int AnimationDuration = 10;
    private const int ImageVariants = 1;

    private NeuralNetwork<Architecture> _network;
    private IModel _model;
    private IEnumerator<NeuralMatrix> _matrixes;
    private bool _hasData;
    private Bitmap _backBuffer;
    private Graphics _backGraphics;

    private readonly Stopwatch _animationWatch = new();
    private readonly Dictionary<string, float[]> _imagesDict = new()
    {
        {"greenEagle256", GraphicsUtils.LoadPixels(@"C:\Users\Seba\Documents\Desktop\Central-Nic-Http-master\greenEagle256.png")},
        {"blueWolf256", GraphicsUtils.LoadPixels(@"C:\Users\Seba\Documents\Desktop\Central-Nic-Http-master\blueWolf256.png")}
    };

    private float AnimationProgress { get; set; }
    private float[] BitMapValues { get; set; }

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

        _backBuffer = new Bitmap(BitmapWidth * ScaleFactor, BitmapHeight * ScaleFactor, PixelFormat.Format32bppArgb);
        _backGraphics = Graphics.FromImage(_backBuffer);
        _backGraphics.InterpolationMode = InterpolationMode.NearestNeighbor;
        _backGraphics.SmoothingMode = SmoothingMode.None;
    }

    public void Run()
    {
        Show();

        _matrixes = _network.RunEpoch().GetEnumerator();
        _animationWatch.Start();

        while (!IsDisposed)
        {
            Application.DoEvents();

            if(!DoStep())
            {
                break;
            }

            Refresh();
        }
    }

    protected override void OnLoad(EventArgs e)
    {
        Prepare();        
    }

    protected override void OnPaint(PaintEventArgs e)
    {
        if (BitMapValues != null && _backBuffer != null)
        {           
            DrawRGB(BitMapValues);

            e.Graphics.DrawImageUnscaled(_backBuffer, 0, 0);
        }
    }

    protected override void OnPaintBackground(PaintEventArgs e)
    {
        // Do nothing to prevent background flicker
    }

    private void Prepare()
    {
        _model = new BitMapTransformationModel();
        _model.Prepare();

        _network = new NeuralNetworkBuilder<Architecture>(_model)
            .WithArchitecture([128, 64, 64, 32, 16])
            .WithEpochs(1000000)
            .WithBatchSize(BatchSize)
            .WithLearningRate(1e-5f)
            .WithHiddenLayerActivation(ActivationType.LeakyReLU)
            .WithOutputLayerActivation(ActivationType.Sigmoid)
            .WithWeightDecay(1e-5f)
            .WithBeta1(0.9f)
            .WithBeta2(0.999f)
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .Build();
    }

    private bool DoStep()
    {
        _hasData = _matrixes.MoveNext();

        if (!_hasData)
        {
            return false;
        }

        AnimationProgress = GetAnimationProgress();

        if (AnimationProgress > 1)
        {
            AnimationProgress = 2 - AnimationProgress;
        }

        var span = _network.Architecture.MatrixNeurons[0].GetRowSpan(0);
        var normalizedAnimation = AnimationProgress * ImageVariants;
        var index = (int)(normalizedAnimation * span.Length);

        var eaglePixels = _imagesDict["greenEagle256"];
        var wolfPixels = _imagesDict["blueWolf256"];

        eaglePixels[..index].CopyTo(span);
        wolfPixels[index..].CopyTo(span);

        BitMapValues = _network.Forward().GetRowSpan(0).ToArray();

        return true;
    }

    private float GetAnimationProgress()
    {
        float animationProgress = (float)_animationWatch.Elapsed.TotalSeconds;

        if (animationProgress > 2 * AnimationDuration)
        {
            _animationWatch.Restart();
            animationProgress = 0;
        }

        animationProgress /= AnimationDuration;

        return animationProgress;
    }

    private unsafe void DrawGrayscale(float[] values)
    {
        var rect = new Rectangle(0, 0, BitmapWidth * ScaleFactor, BitmapHeight * ScaleFactor);
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
        var rect = new Rectangle(0, 0, BitmapWidth * ScaleFactor, BitmapHeight * ScaleFactor);
        var data = _backBuffer.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format32bppPArgb);

        for (int origY = 0; origY < BitmapHeight; origY++)
        {
            for (int origX = 0; origX < BitmapWidth; origX++)
            {
                int rgbIdx = (origY * BitmapWidth + origX) * 3;
                byte r = (byte)(values[rgbIdx + 0] * 255);
                byte g = (byte)(values[rgbIdx + 1] * 255);
                byte b = (byte)(values[rgbIdx + 2] * 255);

                for (int yOffset = 0; yOffset < ScaleFactor; yOffset++)
                {
                    int displayY = origY * ScaleFactor + yOffset;
                    byte* row = (byte*)data.Scan0 + displayY * data.Stride;

                    for (int xOffset = 0; xOffset < ScaleFactor; xOffset++)
                    {
                        int displayX = origX * ScaleFactor + xOffset;
                        int pixelPos = displayX * 4;

                        row[pixelPos + 0] = b;
                        row[pixelPos + 1] = g;
                        row[pixelPos + 2] = r;
                        row[pixelPos + 3] = 255;
                    }
                }
            }
        }

        _backBuffer.UnlockBits(data);
    }
}
