using ILGPU.Runtime;
using ILGPU;
using System.Diagnostics;

namespace ILGPU_ML.Mnist
{
    public static class MnistNetwork
    {
        public static void Run()
        {
            Console.WriteLine("Loading Mnist Dataset");

            List<(float[] data, float[] label)> trainingData = MnistReader.ReadTrainingData();
            Console.WriteLine($"Loaded {trainingData.Count} training sets.");

            List<(float[] data, float[] label)> testData = MnistReader.ReadTestData();
            Console.WriteLine($"Loaded {testData.Count} testing sets.");

            using Network dNetwork = new Network(false, 1337);
            dNetwork.AddInputLater(trainingData[0].data.Length, trainingData[0].data.Length / 10);
            //dNetwork.AddLayer(40);
            //dNetwork.AddLayer(10);
            dNetwork.AddLayer(trainingData[0].label.Length);

            Console.WriteLine("Networks Initalized");

            float learningRate = 0.1f;
            int numberOfEpochs = 50;

            Stopwatch timer = Stopwatch.StartNew();

            MemoryBuffer1D<float, Stride1D.Dense>[] trainingDataData = new MemoryBuffer1D<float, Stride1D.Dense>[trainingData.Count];
            MemoryBuffer1D<float, Stride1D.Dense>[] trainingOutputData = new MemoryBuffer1D<float, Stride1D.Dense>[trainingData.Count];

            for (int i = 0; i < trainingData.Count; i++)
            {
                trainingDataData[i] = dNetwork.device.Allocate1D(trainingData[i].data);
                trainingOutputData[i] = dNetwork.device.Allocate1D(trainingData[i].label);
            }

            Console.WriteLine("Training Data Uploaded To GPU");

            int[] testingIndecies = Utils.GenerateTrainingOrder(testData.Count);
            Random rng = new Random();
            int testingBatchDivisor = 1;

            for (int epoch = 0; epoch < numberOfEpochs; epoch++)
            {
                dNetwork.TrainGPUWithPreloadedData(trainingDataData, trainingOutputData, learningRate);
                Console.WriteLine($"GPU Epoch {epoch} done.");

                int dNetworkCorrectCounter = 0;

                Utils.Shuffle(rng, testingIndecies);

                for (int x = 0; x < testData.Count / testingBatchDivisor; x++)
                {
                    int i = testingIndecies[x];
                    dNetwork.ForwardPassProcess(true, testData[i].data, true);

                    int expectedOutput = Utils.GetIndexOfMax(testData[i].label);
                    int dNetworkOutput = Utils.GetIndexOfMax(dNetwork.layers.Last().LayerData);

                    if (dNetworkOutput == expectedOutput)
                    {
                        dNetworkCorrectCounter++;
                    }

                    //Console.WriteLine("GPU Network | Output: " + dNetworkOutput + " Expected: " + expectedOutput);
                    //Console.WriteLine();
                }

                Console.WriteLine($"GPU Network | {(float)dNetworkCorrectCounter / (testData.Count / (float)testingBatchDivisor)}");
                Console.WriteLine();

            }

            Console.WriteLine($"GPU Network done in: {timer.Elapsed.TotalSeconds}");
        }
    }
}
