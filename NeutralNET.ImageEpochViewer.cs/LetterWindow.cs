using System.Drawing.Drawing2D;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Test.Data;

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

    public const int CapturedRefreshRate = 2000;
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
        _timer.Tick += (_, _) => HandleTick();

        StartTimer();
    }

    private static Color GetHeatmapColor(float v)
    {
        v = Math.Clamp(v, 0f, 1f);
        Color[] colors = [
            Color.FromArgb(0, 0, 0),
            Color.FromArgb(128, 0, 0),
            Color.FromArgb(255, 0, 0),
            Color.FromArgb(255, 128, 0),
            Color.FromArgb(255, 255, 0),
            Color.FromArgb(255, 255, 255)
        ];

        float scaled = v * (colors.Length - 1);
        int idx = (int)scaled;
        if (idx >= colors.Length - 1) return colors[^1];

        float frac = scaled - idx;
        Color c1 = colors[idx];
        Color c2 = colors[idx + 1];

        int r = (int)(c1.R + (c2.R - c1.R) * frac);
        int g = (int)(c1.G + (c2.G - c1.G) * frac);
        int b = (int)(c1.B + (c2.B - c1.B) * frac);

        return Color.FromArgb(r, g, b);
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

        tabControl.Controls.Add(firstTimeTab);
        tabControl.Controls.Add(realTimeTab);
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
                Margin = new Padding(6),
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
            BackColor = Color.FromArgb(18, 18, 18)
        };

        var leftPanel = new Panel
        {
            Width = CapturedWidth + 20,
            Dock = DockStyle.Left,
            BackColor = Color.FromArgb(18, 18, 18)
        };

        ScreenshotSim.Location = new Point(10, 20);
        leftPanel.Controls.Add(ScreenshotSim);

        var rightMainSplitPanel = new Panel
        {
            Dock = DockStyle.Fill,
            BackColor = Color.FromArgb(18, 18, 18)
        };

        realTimeFlowPanel = new FlowLayoutPanel
        {
            Height = 220,
            Dock = DockStyle.Top,
            AutoScroll = true,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = true,
            Padding = new Padding(10),
            BackColor = Color.FromArgb(18, 18, 18)
        };

        foreach (char targetChar in GraphicsUtils.DefaultLetters)
        {
            var slotPanel = new Panel
            {
                Width = 100,
                Height = 50,
                Margin = new Padding(3),
                BorderStyle = BorderStyle.FixedSingle,
                BackColor = Color.FromArgb(28, 28, 30)
            };

            var slotLbl = new Label
            {
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Segoe UI", 8F, FontStyle.Bold),
                ForeColor = Color.White,
                Text = $"{targetChar}: 0.0%"
            };

            slotPanel.Controls.Add(slotLbl);
            realTimeFlowPanel.Controls.Add(slotPanel);
            _realTimeLetterLabels[targetChar] = slotLbl;
        }

        realTimeConvLayersPanel = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            AutoScroll = true,
            FlowDirection = FlowDirection.TopDown,
            WrapContents = false,
            Padding = new Padding(10),
            BackColor = Color.FromArgb(18, 18, 18)
        };

        rightMainSplitPanel.Controls.Add(realTimeConvLayersPanel);
        rightMainSplitPanel.Controls.Add(realTimeFlowPanel);

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

    private async void HandleTick()
    {
        if (Interlocked.CompareExchange(ref TickState, 1, 0) != 0) return;

        try
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

            ScreenshotSim.Invalidate();

            var px = GraphicsUtils.GetPixels(bmp);
            using var inputMatrix = LetterDataLoader.LoadInputFromScreenshot(bmp, px);

            // Execute heavy network calculations asynchronously on thread pool
            await Task.Run(() =>
            {
                using var output = _network.Forward(inputMatrix);

                var convOutputs = new CnnMatrix[NumConvLayers];
                for (int i = 0; i < NumConvLayers; i++)
                {
                    convOutputs[i] = _network.GetConvLayerOutput(inputMatrix, layerIndex: MinConvLayerIndex + i);
                }

                try
                {
                    unsafe
                    {
                        float* pOutput = output.Pointer;
                        int outputCols = output.UsedColumns;

                        var scores = new Dictionary<char, float>();
                        for (int i = 0; i < outputCols && i < GraphicsUtils.DefaultLetters.Length; i++)
                        {
                            scores[GraphicsUtils.DefaultLetters[i]] = pOutput[i];
                        }

                        bool allInitialized = true;
                        for (int i = 0; i < NumConvLayers; i++)
                        {
                            if (!_convLayersInitialized[i]) allInitialized = false;
                        }

                        if (!allInitialized)
                        {
                            Invoke(new Action(() =>
                            {
                                for (int i = 0; i < NumConvLayers; i++)
                                {
                                    if (!_convLayersInitialized[i])
                                    {
                                        InitializeLayerUI(i, convOutputs[i].Channels, convOutputs[i].Width, convOutputs[i].Height);
                                    }
                                }
                            }));

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

                        // Automatically resumes back on UI thread without BeginInvoke
                        Invoke(new Action(() =>
                        {
                            UpdateUIResults(scores);
                        }));
                    }
                }
                finally
                {
                    for (int i = 0; i < NumConvLayers; i++)
                    {
                        convOutputs[i]?.Dispose();
                    }
                }
            });
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

    private void ProcessConvLayerData(CnnMatrix convOutput, int arrayIndex)
    {
        int numFilters = convOutput.Channels;
        int mapHeight = convOutput.Height;
        int mapWidth = convOutput.Width;

        for (int f = 0; f < numFilters; f++)
        {
            float min = float.MaxValue, max = float.MinValue;
            for (int y = 0; y < mapHeight; y++)
            {
                for (int x = 0; x < mapWidth; x++)
                {
                    float val = convOutput[0, f, y, x];
                    if (val < min) min = val;
                    if (val > max) max = val;
                }
            }

            float range = max - min;
            if (range == 0) range = 1f;

            var targetBmp = _cachedLayerBitmaps[arrayIndex][f];

            for (int y = 0; y < mapHeight; y++)
            {
                for (int x = 0; x < mapWidth; x++)
                {
                    float val = convOutput[0, f, y, x];
                    float normalized = (val - min) / range;
                    targetBmp.SetPixel(x, y, GetHeatmapColor(normalized));
                }
            }
        }
    }

    private void InitializeLayerUI(int arrayIndex, int numFilters, int mapWidth, int mapHeight)
    {
        if (_convLayersInitialized[arrayIndex]) return;

        int actualLayerIndex = MinConvLayerIndex + arrayIndex;
        int maxPerRow = 16;
        int itemWidthSize = 68;
        int panelWidth = Math.Min(numFilters, maxPerRow) * itemWidthSize + 30;

        var layerContainer = new Panel
        {
            Width = panelWidth,
            Height = ((numFilters / maxPerRow) + 1) * 76 + 35,
            BackColor = Color.FromArgb(24, 24, 26),
            Margin = new Padding(4, 4, 4, 12),
            BorderStyle = BorderStyle.FixedSingle
        };

        var titleLbl = new Label
        {
            Text = $"Convolutional Layer {actualLayerIndex + 1} Feature Maps (Real-time)",
            ForeColor = Color.White,
            Font = new Font("Segoe UI", 9F, FontStyle.Bold),
            Location = new Point(8, 4),
            Width = 350,
            Height = 20
        };
        layerContainer.Controls.Add(titleLbl);

        var mapsFlow = new FlowLayoutPanel
        {
            Location = new Point(8, 26),
            Width = panelWidth - 16,
            Height = layerContainer.Height - 34,
            FlowDirection = FlowDirection.LeftToRight,
            AutoScroll = true,
            WrapContents = true,
            BackColor = Color.FromArgb(20, 20, 22)
        };
        layerContainer.Controls.Add(mapsFlow);
        realTimeConvLayersPanel.Controls.Add(layerContainer);

        for (int f = 0; f < numFilters; f++)
        {
            var rawBmp = new Bitmap(mapWidth, mapHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            _cachedLayerBitmaps[arrayIndex].Add(rawBmp);

            var scaledBmp = CreateNearestNeighborImage(rawBmp, 64, 64);

            var pic = new PictureBox
            {
                Width = 64,
                Height = 64,
                SizeMode = PictureBoxSizeMode.Normal,
                Image = scaledBmp,
                BackColor = Color.Black,
                BorderStyle = BorderStyle.FixedSingle,
                Margin = new Padding(2)
            };
            _cachedLayerPicBoxes[arrayIndex].Add(pic);
            mapsFlow.Controls.Add(pic);
        }

        _convLayersInitialized[arrayIndex] = true;
    }

    private void UpdateUIResults(Dictionary<char, float> scores)
    {
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
            var buffer = new (PictureBox Pic, Label Lbl, char TargetChar, Bitmap Bmp, char PredChar, float Conf)[_letterSlots.Count];

            Parallel.For(0, _letterSlots.Count, i =>
            {
                var slot = _letterSlots[i];
                char targetChar = slot.TargetChar;
                var (inputMatrix, displayBmp) = LetterDataLoader.GenerateSampleForUI(targetChar);

                try
                {
                    using NeuralMatrix output = _network.Forward(inputMatrix);

                    unsafe
                    {
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

                        buffer[i] = (slot.Pic, slot.Lbl, targetChar, displayBmp, predictedChar, maxConfidence);
                    }
                }
                finally
                {
                    inputMatrix.Dispose();
                }
            });

            return buffer;
        });

        // 2. Automatically resumes on UI thread via SynchronizationContext - update UI all at once
        foreach (var res in results)
        {
            res.Pic.Image?.Dispose();
            res.Pic.Image = res.Bmp;

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
    }

    private void ShowFeatureMaps(char targetChar)
    {
        var (inputMatrix, _) = LetterDataLoader.GenerateSampleForUI(targetChar);

        try
        {
            using var convOutput = _network.GetConvLayerOutput(inputMatrix, layerIndex: 0);

            int numFilters = convOutput.Channels;
            int mapHeight = convOutput.Height;
            int mapWidth = convOutput.Width;

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

            for (int f = 0; f < numFilters; f++)
            {
                Bitmap rawBmp = new Bitmap(mapWidth, mapHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

                float min = float.MaxValue, max = float.MinValue;
                for (int y = 0; y < mapHeight; y++)
                {
                    for (int x = 0; x < mapWidth; x++)
                    {
                        float val = convOutput[0, f, y, x];
                        if (val < min) min = val;
                        if (val > max) max = val;
                    }
                }

                float range = max - min;
                if (range == 0) range = 1f;

                for (int y = 0; y < mapHeight; y++)
                {
                    for (int x = 0; x < mapWidth; x++)
                    {
                        float val = convOutput[0, f, y, x];
                        float normalized = (val - min) / range;
                        rawBmp.SetPixel(x, y, GetHeatmapColor(normalized));
                    }
                }

                var sharpBmp = CreateNearestNeighborImage(rawBmp, 80, 80);
                rawBmp.Dispose();

                PictureBox pic = new PictureBox
                {
                    Width = 80,
                    Height = 80,
                    SizeMode = PictureBoxSizeMode.Normal,
                    Image = sharpBmp,
                    BackColor = Color.Black,
                    BorderStyle = BorderStyle.FixedSingle,
                    Margin = new Padding(4)
                };
                mapPanel.Controls.Add(pic);
            }

            mapForm.ShowDialog();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Nie można pobrać aktywacji warstwy konwolucyjnej: {ex.Message}",
                "Błąd mapy cech", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
        finally
        {
            inputMatrix.Dispose();
        }
    }
}
