using ILGPU;
using ILGPU.IR.Analyses.ControlFlowDirection;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ILGPU_ML
{
    public class Network
    {
        public Context context;
        public Accelerator device;

        public List<Layer> layers;

        private MemoryBuffer1D<float, Stride1D.Dense> inputBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> outputBuffer;

        private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, dLayer> ForwardPassKernel;
        private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, dLayer> FirstBackwardPassKernel;
        private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, dLayer, dLayer> OtherBackwardPassKernel;
        private Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, dLayer, float> BackPropogationKernel;

        private Random rng = new Random();


        public Network(bool debug, int seed = -1)
        {
            if(seed != -1)
            {
                rng = new Random(seed);
            }
            else
            {
                rng = new Random();
            }

            context = Context.Create(builder => builder.CPU().Cuda().EnableAlgorithms());
            device = context.GetPreferredDevice(preferCPU: debug)
                                      .CreateAccelerator(context);
            layers = new List<Layer>();

            ForwardPassKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, dLayer>(ForwardPass);
            FirstBackwardPassKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, dLayer>(FirstBackwardPass);
            OtherBackwardPassKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, dLayer, dLayer>(OtherBackwardPass);
            BackPropogationKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, dLayer, float>(BackPropogation);

        }

        public void AddInputLater(int inputSize, int layerSize)
        {
            if(layers.Count > 0)
            {
                throw new Exception("Cannot allocate more than one input layers");
            }

            Layer layer = new Layer(inputSize, layerSize);
            inputBuffer = device.Allocate1D<float>(inputSize);
            outputBuffer = device.Allocate1D<float>(layerSize);

            layer.Init(rng);
            layer.InitDeviceBuffers(device);

            layers.Add(layer);
        }

        public void AddLayer(int size)
        {
            if(layers.Count < 1)
            {
                throw new Exception("Must set an input layer first");
            }

            Layer layer = new Layer(layers[layers.Count - 1].layerSize, size);
            layer.Init(rng);
            layer.InitDeviceBuffers(device);

            outputBuffer.Dispose();
            outputBuffer = device.Allocate1D<float>(size);

            layers.Add(layer);
        }

        public void Train(bool GPU, float[][] trainingInput, float[][] trainingOutput, float learningWeight)
        {
            int[] trainingOrder = Utils.GenerateTrainingOrder(trainingInput.Length);
            Utils.Shuffle(rng, trainingOrder);

            for(int x = 0; x < trainingOrder.Length; x++)
            {
                int i = trainingOrder[x];

                ForwardPassProcess(GPU, trainingInput[i]);
                BackwardPassProcess(GPU, trainingInput[i], trainingOutput[i], learningWeight);
            }
        }

        public void ForwardPassProcess(bool GPU, float[] input, bool copyToCPU = false)
        {
            if(GPU)
            {
                inputBuffer.CopyFromCPU(input);

                Layer layer0 = layers[0];
                ForwardPassKernel(layer0.inputSize, inputBuffer, layer0.GetDLayer());

                for (int i = 1; i < layers.Count; i++)
                {
                    Layer layer1 = layers[i];
                    ForwardPassKernel(layer1.layerSize, layer0.dLayerData, layer1.GetDLayer());

                    if (copyToCPU)
                    {
                        device.Synchronize();
                        layer0.CopyBackDeviceBuffers();
                        layer1.CopyBackDeviceBuffers();
                    }

                    layer0 = layers[i];
                }

                device.Synchronize();
            }
            else
            {
                Layer layer0 = layers[0];
                layer0.ForwardPassCPU(input);

                for(int i = 1; i < layers.Count; i++)
                {
                    Layer layer1 = layers[i];
                    layer1.ForwardPassCPU(layer0.LayerData);
                    layer0 = layers[i];
                }
            }
        }

        private void BackwardPassProcess(bool GPU, float[] trainingInput, float[] trainingOutput, float learningWeight)
        {
            if (GPU)
            {
                inputBuffer.CopyFromCPU(trainingInput);
                outputBuffer.CopyFromCPU(trainingOutput);

                MemoryBuffer1D<float, Stride1D.Dense>[] errors = new MemoryBuffer1D<float, Stride1D.Dense>[layers.Count];

                for(int i = 0; i < layers.Count; i++)
                {
                    errors[i] = layers[i].dLayerError;
                }

                FirstBackwardPassKernel(layers[layers.Count - 1].layerSize, outputBuffer, errors[layers.Count - 1], layers[layers.Count - 1].GetDLayer());

                for (int i = layers.Count - 2; i >= 0; i--)
                {
                    OtherBackwardPassKernel(layers[i].layerSize, errors[i], errors[i + 1], layers[i].GetDLayer(), layers[i + 1].GetDLayer());
                }

                for (int i = layers.Count - 1; i >= 0; i--)
                {
                    MemoryBuffer1D<float, Stride1D.Dense> layerData;

                    if (i == 0)
                    {
                        layerData = inputBuffer;
                    }
                    else
                    {
                        layerData = layers[i - 1].dLayerData;
                    }

                    BackPropogationKernel(layers[i].layerSize, errors[i], layerData, layers[i].GetDLayer(), learningWeight);
                }

                device.Synchronize();
            }
            else
            {
                float[][] errors = new float[layers.Count][];

                errors[layers.Count - 1] = layers[layers.Count - 1].BackwardPassCPU(trainingOutput);

                for(int i = layers.Count - 2; i >= 0; i--)
                {
                    errors[i] = layers[i].BackwardPassCPU(errors[i + 1], layers[i + 1].LayerWeights);
                }

                for (int i = layers.Count - 1; i >= 0; i--)
                {
                    float[] layerData;

                    if(i == 0)
                    {
                        layerData = trainingInput;
                    }
                    else
                    {
                        layerData = layers[i - 1].LayerData;
                    }

                    layers[i].BackPropogationCPU(errors[i], layerData, learningWeight);
                }
            }
        }

        private static void ForwardPass(Index1D index, ArrayView1D<float, Stride1D.Dense> input, dLayer layer)
        {
            layer.ForwardPass(index, input);
        }

        private static void FirstBackwardPass(Index1D index, ArrayView1D<float, Stride1D.Dense> output, ArrayView1D<float, Stride1D.Dense> error, dLayer layer)
        {
            layer.FirstBackwardPass(index, error, output);
        }

        private static void OtherBackwardPass(Index1D index, ArrayView1D<float, Stride1D.Dense> errorOutput, ArrayView1D<float, Stride1D.Dense> trainingOutput, dLayer layer0, dLayer layer1)
        {
            layer0.OtherBackwardPass(index, errorOutput, trainingOutput, layer1.LayerWeights);
        }

        private static void BackPropogation(Index1D index, ArrayView1D<float, Stride1D.Dense> error, ArrayView1D<float, Stride1D.Dense> input, dLayer layer, float learningWeight)
        {
            layer.BackPropogation(index, error, input, learningWeight);
        }
    }
}
