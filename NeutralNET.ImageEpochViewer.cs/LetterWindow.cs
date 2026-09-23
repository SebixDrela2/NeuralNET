using System.Diagnostics;
using System.Drawing.Drawing2D;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Test.Data;
using NeutralNET.Utils;

namespace NeutralNET.ImageEpochViewer;

public partial class LetterWindow : Form
{
    private TabControl tabControl;
    private TabPage firstTimeTab;
    private TabPage realTimeTab;

    private FlowLayoutPanel flowPanel;
    private FlowLayoutPanel realTimeFlowPanel;
    private FlowLayoutPanel realTimeConvLayersPanel;
    private readonly System.Windows.Forms.Timer _timer;
    private readonly PictureBox ScreenshotSim;

    public const int CapturedRefreshRate = 150;
    public const int CapturedWidth = (int)(GraphicsUtils.Width * CapturedZoom);
    public const int CapturedHeight = (int)(GraphicsUtils.Height * CapturedZoom);
    public const float CapturedZoom = 1;

    public const int CapturedSourcePosX = 394;
    public const int CapturedSourcePosY = 270;

    private readonly CnnNetwork _network;
    private readonly List<(PictureBox Pic, Label Lbl, char TargetChar)> _letterSlots = [];
    private readonly Dictionary<char, Label> _realTimeLetterLabels = [];

    private const int MinConvLayerIndex = 0;
    private const int MaxConvLayerIndex = 2;
    private const int NumConvLayers = (MaxConvLayerIndex - MinConvLayerIndex) + 1;

    private readonly List<PictureBox>[] _cachedLayerPicBoxes = new List<PictureBox>[NumConvLayers];
    private readonly List<Bitmap>[] _cachedLayerBitmaps = new List<Bitmap>[NumConvLayers];
    private readonly bool[] _convLayersInitialized = new bool[NumConvLayers];

    static (int Min, int Max) MinMax(int a, int b) => (int.Min(a, b), int.Max(a, b));

    public LetterWindow(CnnNetwork network)
    {
        // Console.WriteLine($"{Stopwatch.GetTimestamp()} [ctor; {Thread.CurrentThread.ManagedThreadId}]");
        _network = network;

        for (int i = 0; i < NumConvLayers; i++)
        {
            _cachedLayerPicBoxes[i] = [];
            _cachedLayerBitmaps[i] = [];
            _convLayersInitialized[i] = false;
        }

        InitializeComponent();

        ScreenshotSim = new PictureBox
        {
            Width = CapturedWidth,
            Height = CapturedHeight,
            SizeMode = PictureBoxSizeMode.Zoom,
            Image = new Bitmap(CapturedWidth, CapturedHeight),
            BackColor = Color.Black,
            BorderStyle = BorderStyle.FixedSingle,
            Margin = new Padding(4),
        };

        InitializeCustomLayout();

        _timer = new() { Interval = CapturedRefreshRate };
        _timer.Tick += (_, _) =>
        {
            var tabIndex = tabControl?.SelectedIndex ?? -1;
            Task.Run(() => HandleTick(tabIndex));
        };

        StartTimer();
    }

    private static Bitmap CreateNearestNeighborImage(Bitmap src, int targetWidth, int targetHeight)
    {
        var result = new Bitmap(targetWidth, targetHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        using var g = Graphics.FromImage(result);
        g.InterpolationMode = InterpolationMode.NearestNeighbor;
        g.PixelOffsetMode = PixelOffsetMode.Half;
        g.DrawImage(src, new Rectangle(0, 0, targetWidth, targetHeight), new Rectangle(0, 0, src.Width, src.Height), GraphicsUnit.Pixel);
        return result;
    }

    private void InitializeCustomLayout()
    {
        Width = 1300;
        Height = 850;
        Text = "CNN Letter Recognition (All A-Z Grid & Conv Layers 0-2)";
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Color.FromArgb(18, 18, 18);
        TopMost = true;

        tabControl = new TabControl
        {
            Dock = DockStyle.Fill,
            BackColor = Color.FromArgb(18, 18, 18)
        };

        firstTimeTab = new TabPage("First time")
        {
            BackColor = Color.FromArgb(18, 18, 18)
        };

        realTimeTab = new TabPage("Real time")
        {
            BackColor = Color.FromArgb(18, 18, 18)
        };

        tabControl.Controls.Add(realTimeTab);
        tabControl.Controls.Add(firstTimeTab);
        Controls.Add(tabControl);

        flowPanel = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            AutoScroll = true,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = true,
            Padding = new Padding(12),
            BackColor = Color.FromArgb(18, 18, 18)
        };

        firstTimeTab.Controls.Add(flowPanel);

        foreach (char targetChar in GraphicsUtils.DefaultLetters)
        {
            var itemPanel = new Panel
            {
                Width = 145,
                Height = 175,
                Margin = new Padding(1),
                BorderStyle = BorderStyle.FixedSingle,
                BackColor = Color.FromArgb(28, 28, 30),
                Cursor = Cursors.Hand
            };

            var pic = new PictureBox
            {
                Width = GraphicsUtils.Width,
                Height = GraphicsUtils.Height,
                SizeMode = PictureBoxSizeMode.Zoom,
                Location = new Point((itemPanel.Width - GraphicsUtils.Width) / 2, 8),
                BackColor = Color.Black,
                BorderStyle = BorderStyle.FixedSingle,
                Cursor = Cursors.Hand,
                Image = new Bitmap(GraphicsUtils.Width, GraphicsUtils.Height),
            };

            var lbl = new Label
            {
                Width = 135,
                Height = 55,
                TextAlign = ContentAlignment.MiddleCenter,
                Location = new Point(5, 110),
                Font = new Font("Segoe UI", 9F, FontStyle.Bold),
                ForeColor = Color.White,
                Cursor = Cursors.Hand
            };

            EventHandler clickHandler = (s, e) => ShowFeatureMaps(targetChar);
            itemPanel.Click += clickHandler;
            pic.Click += clickHandler;
            lbl.Click += clickHandler;

            itemPanel.Controls.Add(pic);
            itemPanel.Controls.Add(lbl);
            flowPanel.Controls.Add(itemPanel);

            _letterSlots.Add((pic, lbl, targetChar));
        }

        var realTimeMainLayout = new Panel
        {
            Dock = DockStyle.Fill,
            BackColor = Color.FromArgb(18, 18, 18),
            // AutoSize = true,
        };

        var leftPanel = new Panel
        {
            Width = CapturedWidth + 20,
            Dock = DockStyle.Left,
            BackColor = Color.FromArgb(18, 18, 18)
        };

        ScreenshotSim.Location = new Point(10, 20);
        leftPanel.Controls.Add(ScreenshotSim);

        realTimeFlowPanel = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            AutoScroll = true,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = true,
            Padding = new Padding(10),
            BackColor = Color.FromArgb(18, 18, 18)
        };

        foreach (char targetChar in GraphicsUtils.DefaultLetters)
        {
            var slotLbl = new Label
            {
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Consolas", 8F, FontStyle.Bold),
                ForeColor = Color.White,
                Text = $"{targetChar}: 0.0%"
            };
            var slotPanel = new Panel
            {
                Width = (int)(float.Round((slotLbl.PreferredWidth + 24) / 16.0f) * 16),
                Height = (int)(float.Round((slotLbl.PreferredHeight + 16) / 16.0f) * 16),
                Margin = new Padding(3),
                BorderStyle = BorderStyle.FixedSingle,
                BackColor = Color.FromArgb(28, 28, 30),
                Controls = { slotLbl }
            };

            realTimeFlowPanel.Controls.Add(slotPanel);
            _realTimeLetterLabels[targetChar] = slotLbl;
        }

        realTimeConvLayersPanel = new FlowLayoutPanel()
        {
            Dock = DockStyle.Top,
            FlowDirection = FlowDirection.TopDown,
            // AutoScroll = true,
            WrapContents = false,
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowOnly,
            // WrapContents = false,
            // AutoScroll = true,

            // Padding = new Padding(10),
            // BackColor = Color.FromArgb(0xee, 0xee, 0x22),
            // BackColor = Color.FromArgb(50, 50, 51),
        };

        var rightMainSplitPanel = new Panel
        {
            Dock = DockStyle.Fill,
            BackColor = Color.FromArgb(18, 18, 18),
            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowOnly,
            Controls = { realTimeConvLayersPanel, realTimeFlowPanel },
        };

        realTimeMainLayout.Controls.Add(rightMainSplitPanel);
        realTimeMainLayout.Controls.Add(leftPanel);
        realTimeTab.Controls.Add(realTimeMainLayout);

        // Modern async initialization (fires concurrently, updates UI cleanly on completion)
        _ = InitializeLettersAsync();
    }

    private void StartTimer() => _timer.Start();

    protected override void OnFormClosed(FormClosedEventArgs e)
    {
        _timer?.Stop();
        _timer?.Dispose();
        base.OnFormClosed(e);
    }

    private int TickState = 0;

    private void HandleTick(int tabIndex)
    {
        // Console.WriteLine($"{Stopwatch.GetTimestamp()} [HandleTick; {Thread.CurrentThread.ManagedThreadId}]");
        if (Interlocked.CompareExchange(ref TickState, 1, 0) != 0) return;

        try
        {
            switch (tabIndex)
            {
                case 0:
                    {
                        if (ScreenshotSim.Image is not Bitmap bmp) return;

                        var p1 = new Point(CapturedSourcePosX, CapturedSourcePosY);
                        var p2 = new Point(p1.X + CapturedWidth, p1.Y + CapturedHeight);

                        (p1.X, p2.X) = MinMax(p1.X, p2.X);
                        (p1.Y, p2.Y) = MinMax(p1.Y, p2.Y);
                        var pDiff = new Size(p2.X - p1.X, p2.Y - p1.Y);

                        using (var g = Graphics.FromImage(bmp))
                        {
                            g.CopyFromScreen(p1, Point.Empty, pDiff, CopyPixelOperation.SourceCopy);
                            g.Flush();
                        }

                        using var inputMatrix = LetterDataLoader.LoadInputFromBitmap(bmp);
                        Invoke(() => ScreenshotSim.Invalidate());
                        //
                        var convOutputs = _network.GetConvLayerOutput(inputMatrix);
                        {
                            if (convOutputs.Length != NumConvLayers) throw new InvalidOperationException();
                            try
                            {

                                bool allInitialized = true;
                                for (int i = 0; i < NumConvLayers; i++)
                                {
                                    if (!_convLayersInitialized[i]) allInitialized = false;
                                }

                                if (!allInitialized)
                                {
                                    Invoke(() =>
                                    {
                                        for (int i = 0; i < NumConvLayers; ++i)
                                        {
                                            if (!_convLayersInitialized[i])
                                            {
                                                InitializeLayerUI(i, convOutputs[i].Channels, convOutputs[i].Width, convOutputs[i].Height);
                                            }
                                        }
                                    });

                                    while (true)
                                    {
                                        bool check = true;
                                        for (int i = 0; i < NumConvLayers; i++) if (!_convLayersInitialized[i]) check = false;
                                        if (check) break;
                                        Thread.Sleep(5);
                                    }
                                }

                                for (int i = 0; i < NumConvLayers; i++)
                                {
                                    ProcessConvLayerData(convOutputs[i], i);
                                }
                            }
                            finally
                            {
                                for (int i = 0; i < NumConvLayers; i++)
                                {
                                    convOutputs[i]?.Dispose();
                                }
                            }
                            using var output = _network.Forward(inputMatrix);
                            unsafe
                            {
                                float* pOutput = output.Pointer;
                                int outputCols = output.UsedColumns;

                                var scores = new Dictionary<char, float>();
                                for (int i = 0; i < outputCols && i < GraphicsUtils.DefaultLetters.Length; i++)
                                {
                                    scores[GraphicsUtils.DefaultLetters[i]] = pOutput[i];
                                }

                                Invoke(() => UpdateUIResults(scores));
                            }
                        }
                    }
                    break;
                case 1:
                    {

                    }
                    break;
                default:
                    break;
            }
        }
        catch (Exception)
        {
            // Suppress background thread telemetry errors
        }
        finally
        {
            Volatile.Write(ref TickState, 0);
        }
    }

    private unsafe static void ApplyColorScale(CnnMatrix input, int channel, Bitmap output)
    {
        var size = (input.Width, input.Height);
        Debug.Assert((output.Width, output.Height) == size);

        var range = (Min: float.MaxValue, Max: float.MinValue);
        for (int y = 0; y < size.Height; ++y)
        {
            for (int x = 0; x < size.Width; ++x)
            {
                float v = input[0, channel, y, x];
                range = (float.Min(range.Min, v), float.Max(range.Max, v));
            }
        }
        var fr = ((range.Max - range.Min) is var diff and not 0)
            ? float.ReciprocalEstimate(diff)
            : 1;

        // var targetBmp = _cachedLayerBitmaps[arrayIndex][i];
        // Debug.Assert((targetBmp.Width, targetBmp.Height) == (mapWidth, mapHeight));

        var data = output.LockBits(
            new Rectangle(0, 0, size.Width, size.Height),
            System.Drawing.Imaging.ImageLockMode.WriteOnly,
            System.Drawing.Imaging.PixelFormat.Format32bppArgb
        );

        try
        {
            var ptr = (Color32bppArgb*)(void*)data.Scan0;
            for (int y = 0; y < size.Height; ++y, ptr = (Color32bppArgb*)(((byte*)ptr) + data.Stride))
            {
                for (int x = 0; x < size.Width; ++x)
                {
                    ptr[x] = ColorScale.Turbo((input[0, channel, y, x] - range.Min) * fr);
                }
            }

        }
        finally
        {
            output.UnlockBits(data);
        }
    }

    private void ProcessConvLayerData(CnnMatrix convOutput, int arrayIndex)
    {
        // Console.WriteLine($"{Stopwatch.GetTimestamp()} [ProcessConvLayerData; {Thread.CurrentThread.ManagedThreadId}]");
        int n = convOutput.Channels;

        for (int i = 0; i < n; ++i)
        {
            ApplyColorScale(convOutput, i, _cachedLayerBitmaps[arrayIndex][i]);
        }
    }

    private void InitializeLayerUI(int arrayIndex, int numFilters, int mapWidth, int mapHeight)
    {
        // Console.WriteLine($"{Stopwatch.GetTimestamp()} [InitializeLayerUI; {Thread.CurrentThread.ManagedThreadId}]");
        if (_convLayersInitialized[arrayIndex]) return;

        int actualLayerIndex = MinConvLayerIndex + arrayIndex;
        // int maxPerRow = 16;
        // int itemWidthSize = 68;
        // int panelWidth = (int.Min(numFilters, maxPerRow) * itemWidthSize) + 30;
        const int picBoxMargin = 2;
        const int pxStackPadding = 2;
        const int picBoxGap = picBoxMargin * 2;
        const int picBoxSize = 64;

        var hstackPanel = new FlowLayoutPanel()
        {
            Dock = DockStyle.Fill,
            // BackColor = Color.FromArgb(0x33, 0x33, 0xFF),
            FlowDirection = FlowDirection.LeftToRight,

            AutoSize = true,
            WrapContents = true,
            AutoSizeMode = AutoSizeMode.GrowOnly,
            // WrapContents = true,
            // AutoSizeMode = AutoSizeMode.GrowAndShrink,

            MinimumSize = new()
            {
                Width = (16 * picBoxSize) + (15 * picBoxGap) + (2 * pxStackPadding),
                Height = (1 * picBoxSize) + (0 * picBoxGap),
            },
            MaximumSize = new()
            {
                Width = (16 * picBoxSize) + (15 * picBoxGap) + (2 * pxStackPadding),
                Height = (4 * picBoxSize) + (3 * picBoxGap),
            },
        };
        for (int f = 0; f < numFilters; f++)
        {
            var rawBmp = new Bitmap(mapWidth, mapHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            _cachedLayerBitmaps[arrayIndex].Add(rawBmp);

            var scaledBmp = CreateNearestNeighborImage(rawBmp, picBoxSize, picBoxSize);

            var pic = new PictureBox
            {
                Width = picBoxSize,
                Height = picBoxSize,
                SizeMode = PictureBoxSizeMode.Normal,
                Image = scaledBmp,
                BackColor = Color.Black,
                BorderStyle = BorderStyle.FixedSingle,
                Margin = new Padding(picBoxMargin),
            };
            //if (f != 0 && ((f & 0b111) == 0))
            _cachedLayerPicBoxes[arrayIndex].Add(pic);
            hstackPanel.Controls.Add(pic);
        }

        var gBox = new GroupBox()
        {
            Dock = DockStyle.Bottom,
            Text = $"Convolutional Layer {actualLayerIndex + 1}",
            Font = new Font("Consolas", 8, FontStyle.Bold),

            ForeColor = Color.White,
            // BackColor = Color.FromArgb(0x33, 0xFF, 0x33),
            BackColor = Color.FromArgb(50, 50, 51),

            AutoSize = true,
            AutoSizeMode = AutoSizeMode.GrowAndShrink,
            // Anchor = AnchorStyles.Left | AnchorStyles.Right,

            // Dock = DockStyle.Fill,
            // Width = (16 * 64) + (15 * 2),
            Controls = { hstackPanel },
        };


        realTimeConvLayersPanel.Controls.Add(gBox);
        _convLayersInitialized[arrayIndex] = true;
    }

    private void UpdateUIResults(Dictionary<char, float> scores)
    {
        // Console.WriteLine($"{Stopwatch.GetTimestamp()} [UpdateUIResults; {Thread.CurrentThread.ManagedThreadId}]");
        foreach (var kvp in scores)
        {
            char c = kvp.Key;
            float val = kvp.Value;

            if (_realTimeLetterLabels.TryGetValue(c, out var lbl))
            {
                lbl.Text = $"{c}: {val * 100:F1}%";
                if (val > 0.5f)
                {
                    lbl.ForeColor = Color.LightGreen;
                }
                else if (val > 0.2f)
                {
                    lbl.ForeColor = Color.Gold;
                }
                else
                {
                    lbl.ForeColor = Color.Gray;
                }
            }
        }

        for (int i = 0; i < NumConvLayers; i++)
        {
            for (int f = 0; f < _cachedLayerBitmaps[i].Count; f++)
            {
                var rawBmp = _cachedLayerBitmaps[i][f];
                var pic = _cachedLayerPicBoxes[i][f];

                var oldImg = pic.Image;
                pic.Image = CreateNearestNeighborImage(rawBmp, pic.Width, pic.Height);
                oldImg?.Dispose();
            }
        }
    }

    private async Task InitializeLettersAsync()
    {
        // 1. Process all network inferences concurrently in the ThreadPool
        var results = await Task.Run(() =>
        {
            // Console.WriteLine($"{Stopwatch.GetTimestamp()} [InitializeLettersAsync->Task.Run; {Thread.CurrentThread.ManagedThreadId}]");
            var buffer = new (PictureBox Pic, Label Lbl, char TargetChar, char PredChar, float Conf)[_letterSlots.Count];

            unsafe void LoopBody(int i)
            {
                // Console.WriteLine($"{Stopwatch.GetTimestamp()} [InitializeLettersAsync->LoopBody]; {Thread.CurrentThread.ManagedThreadId}]");
                var slot = _letterSlots[i];
                if (slot.Pic.Image is not Bitmap bmp) throw new InvalidOperationException();
                using var inputMatrix = LetterDataLoader.GenerateSampleForUI(slot.TargetChar, bmp);

                {
                    using NeuralMatrix output = _network.Forward(inputMatrix);
                    int predictedClassIndex = 0;
                    float maxConfidence = float.MinValue;

                    float* pOutput = output.Pointer;
                    int outputCols = output.UsedColumns;

                    for (int c = 0; c < outputCols; c++)
                    {
                        float val = pOutput[c];
                        if (val > maxConfidence)
                        {
                            maxConfidence = val;
                            predictedClassIndex = c;
                        }
                    }

                    char predictedChar = (predictedClassIndex >= 0 && predictedClassIndex < GraphicsUtils.DefaultLetters.Length)
                        ? GraphicsUtils.DefaultLetters[predictedClassIndex]
                        : '?';

                    buffer[i] = (slot.Pic, slot.Lbl, slot.TargetChar, predictedChar, maxConfidence);
                }
            }
            for (int i = 0; i < _letterSlots.Count; ++i) LoopBody(i);
            // Parallel.For(0, _letterSlots.Count, LoopBody);

            return buffer;
        });

        Invoke(() =>
        {
            // Console.WriteLine($"{Stopwatch.GetTimestamp()} [InitializeLettersAsync->Invoke; {Thread.CurrentThread.ManagedThreadId}]");
            foreach (var res in results)
            {
                // res.Pic.Image?.Dispose();
                // res.Pic.Image = res.Bmp;
                res.Pic.Invalidate();

                if (res.PredChar == res.TargetChar && res.Conf >= 0.7f)
                {
                    res.Lbl.Text = $"[{res.TargetChar}] Pred: {res.PredChar}\n({res.Conf * 100:F1}%)";
                    res.Lbl.ForeColor = Color.LightGreen;
                }
                else if (res.PredChar != res.TargetChar)
                {
                    res.Lbl.Text = $"[{res.TargetChar}] Pred: {res.PredChar}\n({res.Conf * 100:F1}%)";
                    res.Lbl.ForeColor = Color.IndianRed;
                }
                else
                {
                    res.Lbl.Text = $"[{res.TargetChar}] Pred: {res.PredChar}\n({res.Conf * 100:F1}%)";
                    res.Lbl.ForeColor = Color.Gold;
                }
            }
        });
    }

    private void ShowFeatureMaps(char targetChar)
    {
        // Console.WriteLine($"{Stopwatch.GetTimestamp()} [ShowFeatureMaps; {Thread.CurrentThread.ManagedThreadId}]");
        var (inputMatrix, bmp) = LetterDataLoader.GenerateSampleForUI(targetChar);
        bmp.Dispose();

        try
        {
            Form mapForm = new Form
            {
                Width = 720,
                Height = 540,
                Text = $"Convolutional Layer 1 Feature Maps for '{targetChar}'",
                StartPosition = FormStartPosition.CenterParent,
                BackColor = Color.FromArgb(18, 18, 18),
                TopMost = true
            };

            FlowLayoutPanel mapPanel = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoScroll = true,
                FlowDirection = FlowDirection.LeftToRight,
                Padding = new Padding(10),
                BackColor = Color.FromArgb(18, 18, 18)
            };
            mapForm.Controls.Add(mapPanel);

            var convOutput = _network.GetConvLayerOutput(inputMatrix);
            try
            {

                int numFilters = convOutput[0].Channels;
                int mapHeight = convOutput[0].Height;
                int mapWidth = convOutput[0].Width;


                for (int f = 0; f < numFilters; f++)
                {
                    using Bitmap rawBmp = new Bitmap(mapWidth, mapHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                    ApplyColorScale(convOutput[0], f, rawBmp);

                    mapPanel.Controls.Add(new PictureBox
                    {
                        Width = 80,
                        Height = 80,
                        SizeMode = PictureBoxSizeMode.Normal,
                        Image = CreateNearestNeighborImage(rawBmp, 80, 80),
                        BackColor = Color.Black,
                        BorderStyle = BorderStyle.FixedSingle,
                        Margin = new Padding(4)
                    });
                }

            }
            finally
            {
                convOutput.DisposeEach();
            }

            mapForm.ShowDialog();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Something is no yes: {ex.Message}", "Yesn't", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
        finally
        {
            inputMatrix.Dispose();
        }
    }
}

public static class ColorScale
{
    public static (byte R, byte G, byte B) Turbo(float value)
    {
        ReadOnlySpan<float> r = stackalloc float[] { 0.13572138f * 0xFF, 4.615392600f * 0xFF, -42.66032258f * 0xFF, 132.13108234f * 0xFF, -152.94239396f * 0xFF, 59.28637943f * 0xFF, 0, 0 };
        ReadOnlySpan<float> g = stackalloc float[] { 0.09140261f * 0xFF, 2.194188390f * 0xFF, 4.8429665800f * 0xFF, -14.18503333f * 0xFF, 4.27729857000f * 0xFF, 2.829566040f * 0xFF, 0, 0 };
        ReadOnlySpan<float> b = stackalloc float[] { 0.10667330f * 0xFF, 12.64194608f * 0xFF, -60.58204836f * 0xFF, 110.36276771f * 0xFF, -89.903109120f * 0xFF, 27.34824973f * 0xFF, 0, 0 };
        Span<float> x = stackalloc float[8];
        x.Clear();

        x[0] = 1;
        x[1] = float.Clamp(value, 0, 1);
        x[2] = x[1] * x[1];
        x[3] = x[2] * x[1];
        x[4] = x[2] * x[2];
        x[5] = x[1] * x[4];

        return (
            byte.CreateSaturating((r[0] * x[0]) + (r[1] * x[1]) + (r[2] * x[2]) + (r[3] * x[3]) + (r[4] * x[4]) + (r[5] * x[5]) + 0.5f),
            byte.CreateSaturating((g[0] * x[0]) + (g[1] * x[1]) + (g[2] * x[2]) + (g[3] * x[3]) + (g[4] * x[4]) + (g[5] * x[5]) + 0.5f),
            byte.CreateSaturating((b[0] * x[0]) + (b[1] * x[1]) + (b[2] * x[2]) + (b[3] * x[3]) + (b[4] * x[4]) + (b[5] * x[5]) + 0.5f)
        );
    }
}
