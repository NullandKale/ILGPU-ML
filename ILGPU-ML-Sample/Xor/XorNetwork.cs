using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ILGPU_ML.Xor
{
    public static class XorNetwork
    {
        public static void Run()
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

            using Network network = new Network(false, 1337);
            network.AddInputLater(2, 2);
            network.AddLayer(1);

            using Network dNetwork = new Network(false, 1337);
            dNetwork.AddInputLater(2, 2);
            dNetwork.AddLayer(1);

            float learningRate = 0.1f;
            int numberOfEpochs = 100000;

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
