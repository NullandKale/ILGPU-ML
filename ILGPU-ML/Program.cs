namespace ILGPU_ML
{
    public struct Layer
    {
        public int input;
        public int size;

        public float[][] LayerWeights;
        public float[] LayerData;
        public float[] LayerBias;

        public Layer(int input, int size)
        {
            this.input = input;
            this.size = size;
            LayerWeights = new float[input][];
                
            for(int i = 0; i < input; i++)
            {
                LayerWeights[i] = new float[size];
            }

            LayerData = new float[size];
            LayerBias = new float[size];
        }

        public void Init(int seed)
        {
            Random rng = new Random(seed);

            for (int i = 0; i < input; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    LayerWeights[i][j] = (float)rng.NextDouble();

                    if (i == 0)
                    {
                        LayerData[j] = (float)rng.NextDouble();
                        LayerBias[j] = (float)rng.NextDouble();
                    }
                }
            }
        }

        public void ForwardPass(float[] trainingInput)
        {
            for (int j = 0; j < size; j++)
            {
                float activation = LayerBias[j];

                for (int k = 0; k < input; k++)
                {
                    activation += trainingInput[k] * LayerWeights[k][j];
                }

                LayerData[j] = Network.sigmoid(activation);
            }
        }

        public float[] BackwardPass(float[] trainingOutput)
        {
            float[] deltaOutput = new float[size];

            for (int j = 0; j < size; j++)
            {
                float error = (trainingOutput[j] - LayerData[j]);
                deltaOutput[j] = error * Network.dSigmoid(LayerData[j]);
            }

            return deltaOutput;
        }

        public float[] BackwardPass(float[] trainingOutput, float[][] outputWeights)
        {
            float[] deltaOutput = new float[size];

            for (int j = 0; j < size; j++)
            {
                float error = 0.0f;

                for(int k = 0; k < trainingOutput.Length; k++)
                {
                    error += trainingOutput[k] * outputWeights[j][k];
                }

                deltaOutput[j] = error * Network.dSigmoid(LayerData[j]);
            }

            return deltaOutput;
        }

        public void BackPropogation(float[] outputError, float[] inputLayerData, float learningWeight)
        {
            for (int j = 0; j < size; j++)
            {
                LayerBias[j] += outputError[j] * learningWeight;
                for (int k = 0; k < input; k++)
                {
                    LayerWeights[k][j] += inputLayerData[k] * outputError[j] * learningWeight;
                }
            }
        }
    }

    public struct Network
    {
        public const int numInputs = 2;
        public const int numHiddenNodes = 2;
        public const int numOutputs = 1;

        public Layer input = new Layer(numInputs, numHiddenNodes);
        public Layer output = new Layer(numHiddenNodes, numOutputs);

        public Network()
        {

        }

        public void Init(int seed)
        {
            input.Init(seed);
            output.Init(seed);
        }

        public void Train(Random rng, float[][] trainingInput, float[][] trainingOutput, float learningWeight)
        {
            int[] trainingOrder = new int[] { 0, 1, 2, 3 };
            Shuffle(rng, trainingOrder);

            for (int x = 0; x < trainingOrder.Length; x++)
            {
                int i = trainingOrder[x];

                ForwardPass(trainingInput[i]);
                BackwardPass(trainingInput[i], trainingOutput[i], learningWeight);
            }
        }

        public void ForwardPass(float[] trainingInput)
        {
            input.ForwardPass(trainingInput);
            output.ForwardPass(input.LayerData);
        }

        public void BackwardPass(float[] trainingInput, float[] trainingOutput, float learningWeight)
        {
            float[] outputError = output.BackwardPass(trainingOutput);
            float[] inputError = input.BackwardPass(outputError, output.LayerWeights);
            output.BackPropogation(outputError, input.LayerData, learningWeight);
            input.BackPropogation(inputError, trainingInput, learningWeight);
        }

        public static float sigmoid(float x)
        {
            return 1f / (1f + MathF.Exp(-x));
        }

        public static float dSigmoid(float x)
        {
            return x * (1 - x);
        }

        public static void Shuffle<T>(Random rng, T[] array)
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }
    }

    internal class Program
    {
        const int numTrainingSets = 4;

        static float[][] trainingInputs;
        static float[][] trainingOutputs;

        static void Main(string[] args)
        {
            trainingInputs = new float[][] { new float[] { 0f, 0f },
                                             new float[] { 1f, 0f },
                                             new float[] { 0f, 1f },
                                             new float[] { 1f, 1f}};

            trainingOutputs = new float[][] { new float[] { 0f },
                                              new float[] { 1f },
                                              new float[] { 1f },
                                              new float[] { 0f }};

            Network nn = new Network();
            nn.Init(1337);

            float learningRate = 0.1f;
            int numberOfEpochs = 10000;
            Random rng = new Random();

            for(int epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                nn.Train(rng, trainingInputs, trainingOutputs, learningRate);
            }

            for (int i = 0; i < numTrainingSets; i++)
            {
                nn.ForwardPass(trainingInputs[i]);
                Console.WriteLine("Input: [" + trainingInputs[i][0] + ", " + trainingInputs[i][1] + "] Output: " + nn.output.LayerData[0] + " Expected: " + trainingOutputs[i][0]);

            }

        }
    }
}