namespace ILGPU_ML
{
    public struct Network
    {
        public const int numInputs = 2;
        public const int numHiddenNodes = 2;
        public const int numOutputs = 1;

        public float[,] HiddenLayerWeights = new float[numInputs,numHiddenNodes];
        public float[] HiddenLayer = new float[numHiddenNodes];
        public float[] HiddenLayerBias = new float[numHiddenNodes];

        public float[,] OutputLayerWeights = new float[numHiddenNodes, numOutputs];
        public float[] OutputLayer = new float[numOutputs];
        public float[] OutputLayerBias = new float[numOutputs];

        public Network()
        {

        }
    }

    internal class Program
    {
        const int numTrainingSets = 4;

        static float[,] trainingInputs = new float[Network.numInputs, numTrainingSets];

        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");

            trainingInputs = new 

            double learningRate = 0.1;


        }
    }
}