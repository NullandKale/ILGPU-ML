using System.Diagnostics;

namespace ILGPU_ML
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //XorSample();
            MnistSample();
        }

        static void MnistSample()
        {
            Console.WriteLine("Loading Mnist Dataset");
            
            List<(float[] data, float[] label)> trainingData = MnistReader.ReadTrainingData();
            Console.WriteLine($"Loaded {trainingData.Count} training sets.");

            List<(float[] data, float[] label)> testData = MnistReader.ReadTestData();
            Console.WriteLine($"Loaded {testData.Count} testing sets.");

            Network dNetwork = new Network(false, 1337);
            dNetwork.AddInputLater(trainingData[0].data.Length, trainingData[0].data.Length);
            dNetwork.AddLayer(200);
            dNetwork.AddLayer(30);
            dNetwork.AddLayer(trainingData[0].label.Length);

            Console.WriteLine("Networks Initalized");

            float learningRate = 0.01f;
            int numberOfEpochs = 20;

            Stopwatch timer = Stopwatch.StartNew();

            int[] trainingOrder = Utils.GenerateTrainingOrder(trainingData.Count);
            Random rng = new Random();

            for (int epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                Utils.Shuffle(rng, trainingOrder);
                for (int x = 0; x < trainingData.Count; x++)
                {
                    int i = trainingOrder[x];
                    dNetwork.Train(true, trainingData[i].data, trainingData[i].label, learningRate);
                }
                Console.WriteLine($"GPU Epoch {epoch} done.");
            }

            Console.WriteLine($"GPU Network done in: {timer.Elapsed.TotalSeconds}");

            int dNetworkCorrectCounter = 0;

            for (int i = 0; i < testData.Count; i++)
            {
                dNetwork.ForwardPassProcess(true, testData[i].data, true);

                int expectedOutput = Utils.GetIndexOfMax(testData[i].label);
                int dNetworkOutput = Utils.GetIndexOfMax(dNetwork.layers.Last().LayerData);

                if(dNetworkOutput == expectedOutput)
                {
                    dNetworkCorrectCounter++;
                }

                //Console.WriteLine("GPU Network | Output: " + dNetworkOutput + " Expected: " + expectedOutput);
                //Console.WriteLine();
            }

            Console.WriteLine($"GPU Network | {(float)dNetworkCorrectCounter / testData.Count}");
            Console.WriteLine();

        }

        static void XorSample()
        {
            int numTrainingSets = 4;

            float[][] trainingInputs = new float[][] { new float[] { 0f, 0f },
                                                       new float[] { 1f, 0f },
                                                       new float[] { 0f, 1f },
                                                       new float[] { 1f, 1f}};

            float[][] trainingOutputs = new float[][] { new float[] { 0f },
                                                        new float[] { 1f },
                                                        new float[] { 1f },
                                                        new float[] { 0f }};

            Network network = new Network(true, 1337);
            network.AddInputLater(2, 2);
            network.AddLayer(1);

            Network dNetwork = new Network(false, 1337);
            dNetwork.AddInputLater(2, 2);
            dNetwork.AddLayer(1);

            float learningRate = 0.1f;
            int numberOfEpochs = 10000;

            Console.WriteLine("Networks Initalized");

            for (int epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                network.Train(false, trainingInputs, trainingOutputs, learningRate);
            }
            Console.WriteLine("CPU Network done.");

            for (int epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                dNetwork.Train(true, trainingInputs, trainingOutputs, learningRate);
            }
            Console.WriteLine("GPU Network done.");

            for (int i = 0; i < numTrainingSets; i++)
            {
                network.ForwardPassProcess(false, trainingInputs[i]);
                dNetwork.ForwardPassProcess(true, trainingInputs[i], true);

                Console.WriteLine("CPU Network | Input: [" + trainingInputs[i][0] + ", " + trainingInputs[i][1] + "] Output: " + network.layers[1].LayerData[0] + " Expected: " + trainingOutputs[i][0]);
                Console.WriteLine("GPU Network | Input: [" + trainingInputs[i][0] + ", " + trainingInputs[i][1] + "] Output: " + dNetwork.layers[1].LayerData[0] + " Expected: " + trainingOutputs[i][0]);
                Console.WriteLine();
            }
        }
    }
}