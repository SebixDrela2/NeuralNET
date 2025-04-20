using NeutralNET.Stuff;

namespace NeutralNET.Matrices;

public struct Matrix
{
    public int Rows;
    public int Columns;
    public ArraySegment<float> Data;
    public readonly float FirstElement => Data[0];

    public Matrix(int rows, int columns)
    {
        Rows = rows;
        Columns = columns;     
        Data = new float[rows * columns];
    }

    public void ApplySigmoid()
    {
        for (var row = 0; row < Rows; row++)
        {
            for (var column = 0; column < Columns; column++)
            {
                Set(row, column, MathUtils.Sigmoid(At(row, column)));
            }
        }
    }

    public Matrix Dot(in Matrix other)
    {
        if (Columns != other.Rows)
        {
            throw new ArgumentException($"Rows of current: {Rows} do not match other Columns {other.Columns}");
        }

        var innerColumnSize = Columns;
        var result = new Matrix(Rows, other.Columns);

        for (var row = 0; row < result.Rows; row++)
        {
            for (var column = 0; column < result.Columns; column++)
            {
                result.Set(row, column, 0);

                for (var k = 0 ; k < innerColumnSize; k++)
                {
                    var innerAt = At(row, k);
                    var outerAt = other.At(k, column);

                    var multipliedResult = innerAt * outerAt;
                    result.Add(row, column, multipliedResult);
                }              
            }
        }

        return result;
    }

    public void CopyDataFrom(Matrix other)
    {
        if (Rows != other.Rows)
        {
            throw new ArgumentException($"Rows of current: {Rows} do not rows match other Rows: {other.Rows}");
        }

        if (Columns != other.Columns)
        {
            throw new ArgumentException($"Columns of current: {Columns} do not match other Columns: {other.Columns}");
        }

        for (var row = 0; row < other.Rows; row++)
        {
            for (var column = 0; column < other.Columns; column++)
            {
                Set(row, column, other.At(row, column));
            }
        }
    }

    public Matrix SplitStart(int column)
    {
        var result = new List<float>();

        var rows = Rows;
        var columns = Columns;

        for (var i = 0; i < Rows; i++)
        {
            var row = Row(i);

            result.AddRange(row.Data.Take(column));
        }

        return new Matrix(rows, column)
        {
            Data = new ArraySegment<float>(result.ToArray())
        };
    }

    public Matrix SplitEnd(int column)
    {
        var result = new List<float>();

        var rows = Rows;
        var columns = Columns;

        for (var i = 0; i < Rows; i++)
        {
            var row = Row(i);

            result.AddRange(row.Data.Skip(column - 1).Take(1));
        }

        return new Matrix(rows, 1)
        {
            Data = new ArraySegment<float>(result.ToArray())
        };
    }

    public Matrix Row(int row)
    {
        var result = new Matrix(1, Columns)
        {           
            Data = Data.Slice(Columns * row, Columns)
        };

        return result;
    }

    public void Sum(in Matrix other)
    {
        if (Rows != other.Rows)
        {
            throw new ArgumentException($"Rows of current: {Rows} do not rows match other Rows: {other.Rows}");
        }

        if (Columns != other.Columns)
        {
            throw new ArgumentException($"Columns of current: {Rows} do not match other Columns: {other.Rows}");
        }

        for (var row = 0; row < Rows; row++)
        {
            for (var column = 0; column < Columns; column++)
            {
                Add(row, column, other.At(row, column));
            }
        }
    }

    public readonly float At(int row, int column) => Data[(row * Columns) + column];
    
    public void Randomize(float low = 0, float high = 1)
    {
        for (var i = 0; i < Rows; i++)
        {
            for (var j = 0; j < Columns; j++)
            {
                Set(i, j, RandomUtils.GetFloat(1) * (high - low) + low);
            }            
        }
    }

    public void Fill(float value)
    {
        for (var i = 0; i < Rows; i++)
        {
            for (var j = 0; j < Columns; j++)
            {
                Set(i, j, value);
            }
        }
    }

    public readonly void Print(string name)
    {
        Console.WriteLine($"{name} = [");

        for (var i = 0; i < Rows; i++)
        {
            for (var j = 0; j < Columns; j++)
            {
                var value = At(i, j);
                Console.Write($"    {value:f5}");
            }

            Console.SetCursorPosition(0, Console.CursorTop);
            Console.WriteLine();
        }

        Console.WriteLine($"]");
    }

    public void Set(int row, int column, float value)
    {
        Data[(row * Columns) + column] = value;
    }

    public void Add(int row, int column, float value)
    {
        Data[(row * Columns) + column] += value;
    }

    public void Sub(int row, int column, float value)
    {
        Data[(row * Columns) + column] -= value;
    }

    public void Divide(int row, int column, float value)
    {
        Data[(row * Columns) + column] /= value;
    }
}
