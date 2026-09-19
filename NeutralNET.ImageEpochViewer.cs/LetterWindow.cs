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
    private readonly System.Windows.Forms.Timer _timer;
    private readonly PictureBox DebugPreview;

    public const int CapturedRefreshRate = 750;
    public const int CapturedWidth = (int)(GraphicsUtils.Width * CapturedZoom);
    public const int CapturedHeight = (int)(GraphicsUtils.Height * CapturedZoom);
    public const float CapturedZoom = 1;

    public const int CapturedSourcePosX = 1545;
    public const int CapturedSourcePosY = 270;

    private readonly CnnNetwork _network;
    private readonly List<(PictureBox Pic, Label Lbl, char TargetChar)> _letterSlots = [];


    static (int Min, int Max) MinMax(int a, int b) => (int.Min(a, b), int.Max(a, b));

    public LetterWindow(CnnNetwork network)
    {
        _network = network;

        InitializeComponent();
        DebugPreview = new PictureBox
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

    private void InitializeCustomLayout()
    {
        Width = 1040;
        Height = 720;
        Text = "CNN Letter Recognition (All A-Z Grid - Click any letter to inspect Conv Layers)";
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Color.FromArgb(18, 18, 18);

        // Tworzenie głównego kontenera zakładek
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

        // Dodanie siatki liter do zakładki "First time"
        firstTimeTab.Controls.Add(flowPanel);

        foreach (char targetChar in GraphicsUtils.DefaultLetters)
        {
            var itemPanel = new Panel
            {
                Width = 145,
                Height = 175,
                Margin = new Padding(6),
                BorderStyle = BorderStyle.FixedSingle,
                BackColor = Color.FromArgb(28, 28, 30, 30),
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

        flowPanel.Controls.Add(DebugPreview);
    }

    private void StartTimer() => _timer.Start();

    protected override void OnFormClosed(FormClosedEventArgs e)
    {
        _timer?.Stop();
        _timer?.Dispose();
        base.OnFormClosed(e);
    }

    private void UpdateCapture()
    {
        if (DebugPreview.Image is not Bitmap bmp) throw new InvalidOperationException();

        using (var g = Graphics.FromImage(bmp))
        {
            var p1 = new Point(CapturedSourcePosX, CapturedSourcePosY);
            var p2 = new Point(p1.X - CapturedWidth, p1.Y + CapturedHeight);

            (p1.X, p2.X) = MinMax(p1.X, p2.X);
            (p1.Y, p2.Y) = MinMax(p1.Y, p2.Y);
            var pDiff = new Size(p2.X - p1.X, p2.Y - p1.Y);

            g.CopyFromScreen(p1, Point.Empty, pDiff, CopyPixelOperation.SourceCopy);
            g.Flush();
        }

        var px = GraphicsUtils.GetPixels(bmp); // !! <--- TUTAJ MASZ PIXELE <--- !!
        DebugPreview.Invalidate();
    }

    private void HandleTick()
    {
        UpdateCapture();
        RefreshAllLetters();
    }

    private unsafe void RefreshAllLetters()
    {
        foreach (var slot in _letterSlots)
        {
            char targetChar = slot.TargetChar;
            var (inputMatrix, displayBmp) = LetterDataLoader.GenerateSampleForUI(targetChar);

            slot.Pic.Image?.Dispose();
            slot.Pic.Image = displayBmp;

            using NeuralMatrix output = _network.Forward(inputMatrix);
            inputMatrix.Dispose();

            int predictedClassIndex = 0;
            float maxConfidence = float.MinValue;

            float* pOutput = output.Pointer;
            int outputCols = output.UsedColumns;

            for (int i = 0; i < outputCols; i++)
            {
                float val = pOutput[i];
                if (val > maxConfidence)
                {
                    maxConfidence = val;
                    predictedClassIndex = i;
                }
            }

            char predictedChar = (predictedClassIndex >= 0 && predictedClassIndex < GraphicsUtils.DefaultLetters.Length)
                ? GraphicsUtils.DefaultLetters[predictedClassIndex]
                : '?';

            if (predictedChar == targetChar && maxConfidence >= 0.7f)
            {
                slot.Lbl.Text = $"[{targetChar}] Pred: {predictedChar}\n({maxConfidence * 100:F1}%)";
                slot.Lbl.ForeColor = Color.LightGreen;
            }
            else if (predictedChar != targetChar)
            {
                slot.Lbl.Text = $"[{targetChar}] Pred: {predictedChar}\n({maxConfidence * 100:F1}%)";
                slot.Lbl.ForeColor = Color.IndianRed;
            }
            else
            {
                slot.Lbl.Text = $"[{targetChar}] Pred: {predictedChar}\n({maxConfidence * 100:F1}%)";
                slot.Lbl.ForeColor = Color.Gold;
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
                BackColor = Color.FromArgb(18, 18, 18)
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
                Bitmap mapBmp = new Bitmap(mapWidth, mapHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

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
                        int normalized = (int)(((val - min) / range) * 255f);
                        normalized = Math.Max(0, Math.Min(255, normalized));

                        mapBmp.SetPixel(x, y, Color.FromArgb(normalized, normalized, normalized));
                    }
                }

                PictureBox pic = new PictureBox
                {
                    Width = 80,
                    Height = 80,
                    SizeMode = PictureBoxSizeMode.Zoom,
                    Image = mapBmp,
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
            MessageBox.Show($"Could not retrieve convolutional layer activations: {ex.Message}",
                "Feature Map Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }
        finally
        {
            inputMatrix.Dispose();
        }
    }
}
