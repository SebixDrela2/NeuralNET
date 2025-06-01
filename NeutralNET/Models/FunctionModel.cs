using NeutralNET.Matrices;

namespace NeutralNET.Models;

internal class FunctionModel(Func<int, int, int> func) : IModel
{


    public NeuralMatrix TrainingInput { get ; set ; }
    public NeuralMatrix TrainingOutput { get ; set ; }


    public void Prepare()
    {
       
    }

    private void PrepareFunc()
    {

    }
}
