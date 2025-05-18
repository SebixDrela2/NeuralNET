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
    private readonly System.Windows.Forms.Timer _refreshTimer = new();


    public float[]? BitMapValues => _isTraining ? _matrixes.Current.ToArray() : null;

    public Form1()
    {
        InitializeComponent();

        ClientSize = new Size(BitmapWidth, BitmapHeight);
        StartPosition = FormStartPosition.CenterScreen;

        DoubleBuffered = true;
        SetStyle(ControlStyles.OptimizedDoubleBuffer |
                 ControlStyles.AllPaintingInWmPaint |
                 ControlStyles.UserPaint, true);
        UpdateStyles();

        _backBuffer = new Bitmap(BitmapWidth, BitmapHeight, PixelFormat.Format32bppArgb);
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

            _isTraining = _matrixes.MoveNext();

            if (!_isTraining)
            {
                break;
            }
        }
    }

    protected override void OnLoad(EventArgs e)
    {
        Prepare();
        _matrixes = _network.EnumerateEpochs().GetEnumerator();
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
            .WithArchitecture([32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32])
            .WithEpochs(500)
            .WithBatchSize(BatchSize)
            .WithLearningRate(1e-4f)
            .WithWeightDecay(3e-5f)
            .WithBeta1(0.9f)
            .WithBeta2(0.999f)
            .WithEpsilon(1e-8f)
            .WithShuffle(true)
            .Build();
    }
    
    private unsafe void DrawGrayscale(float[] values)
    {
        var rect = new Rectangle(0, 0, BitmapWidth, BitmapHeight);
        var data = _backBuffer.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format32bppPArgb);

        Parallel.For(0, BitmapHeight, y =>
        {
            byte* row = (byte*)data.Scan0 + y * data.Stride;
            for (int x = 0; x < BitmapWidth; x++)
            {
                int idx = y * BitmapWidth + x;
                byte brightness = (byte)(values[idx] * 255); 

                row[x * 4 + 0] = brightness;
                row[x * 4 + 1] = brightness;
                row[x * 4 + 2] = brightness;
                row[x * 4 + 3] = 255;       
            }
        });

        _backBuffer.UnlockBits(data);
    }
    
    private unsafe void DrawRGB(float[] values)
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
