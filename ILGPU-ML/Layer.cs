using ILGPU;
using ILGPU.Runtime;

namespace ILGPU_ML
{
    public struct dLayer
    {
        public int inputSize;
        public int layerSize;

        public ArrayView2D<float, Stride2D.DenseY> LayerWeights;
        public ArrayView1D<float, Stride1D.Dense> LayerData;
        public ArrayView1D<float, Stride1D.Dense> LayerBias;
        public ArrayView1D<float, Stride1D.Dense> LayerError;

        public dLayer(int inputSize, int layerSize,
                      MemoryBuffer2D<float, Stride2D.DenseY> LayerWeights,
                      MemoryBuffer1D<float, Stride1D.Dense> LayerData,
                      MemoryBuffer1D<float, Stride1D.Dense> LayerBias,
                      MemoryBuffer1D<float, Stride1D.Dense> LayerError)
        {
            this.inputSize = inputSize;
            this.layerSize = layerSize;

            this.LayerWeights = LayerWeights;
            this.LayerData = LayerData;
            this.LayerBias = LayerBias;
            this.LayerError = LayerError;
        }

        public void ForwardPass(int j, ArrayView1D<float, Stride1D.Dense> trainingInput)
        {
            float activation = LayerBias[j];

            for (int k = 0; k < inputSize; k++)
            {
                activation += trainingInput[k] * LayerWeights[k, j];
            }

            LayerData[j] = Utils.lrelu(activation);
        }

        public void FirstBackwardPass(int j, ArrayView1D<float, Stride1D.Dense> errorOutput, ArrayView1D<float, Stride1D.Dense> trainingOutput)
        {
            float error = (trainingOutput[j] - LayerData[j]);
            errorOutput[j] = error * Utils.dlrelu(LayerData[j]);
        }

        public void OtherBackwardPass(int j, ArrayView1D<float, Stride1D.Dense> errorOutput, ArrayView1D<float, Stride1D.Dense> trainingOutput, ArrayView2D<float, Stride2D.DenseY> outputWeights)
        {
            float error = 0.0f;

            for (int k = 0; k < trainingOutput.Length; k++)
            {
                error += trainingOutput[k] * outputWeights[j, k];
            }

            errorOutput[j] = error * Utils.dlrelu(LayerData[j]);
        }

        public void BackPropogation(int j, ArrayView1D<float, Stride1D.Dense> outputError, ArrayView1D<float, Stride1D.Dense> inputLayerData, float learningWeight)
        {
            LayerBias[j] += outputError[j] * learningWeight;
            for (int k = 0; k < inputSize; k++)
            {
                LayerWeights[k, j] += inputLayerData[k] * outputError[j] * learningWeight;
            }
        }
    }


    public class Layer : IDisposable
    {
        public int inputSize;
        public int layerSize;

        public float[][] LayerWeights;
        public float[] LayerData;
        public float[] LayerBias;
        public float[] LayerError;

        public MemoryBuffer2D<float, Stride2D.DenseY> dLayerWeights;
        public MemoryBuffer1D<float, Stride1D.Dense> dLayerData;
        public MemoryBuffer1D<float, Stride1D.Dense> dLayerBias;
        public MemoryBuffer1D<float, Stride1D.Dense> dLayerError;

        public Layer(int input, int size)
        {
            this.inputSize = input;
            this.layerSize = size;
            LayerWeights = new float[input][];
                
            for(int i = 0; i < input; i++)
            {
                LayerWeights[i] = new float[size];
            }

            LayerData = new float[size];
            LayerBias = new float[size];
            LayerError = new float[size];
        }

        public void InitDeviceBuffers(Accelerator device)
        {
            dLayerWeights = device.Allocate2DDenseY(Utils.To2D(LayerWeights));
            dLayerData = device.Allocate1D(LayerData);
            dLayerBias = device.Allocate1D(LayerBias);
            dLayerError = device.Allocate1D(LayerError);
        }

        public void CopyBackDeviceBuffers()
        {
            LayerWeights = Utils.ToJaggedArray(dLayerWeights.GetAsArray2D());
            dLayerData.CopyToCPU(LayerData);
            dLayerBias.CopyToCPU(LayerBias);
            dLayerError.CopyToCPU(LayerError);
        }

        public dLayer GetDLayer()
        {
            return new dLayer(inputSize, layerSize, dLayerWeights, dLayerData, dLayerBias, dLayerError);
        }

        public void Init(Random rng)
        {
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < layerSize; j++)
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

        public void ForwardPassCPU(float[] trainingInput)
        {
            for (int j = 0; j < layerSize; j++)
            {
                float activation = LayerBias[j];

                for (int k = 0; k < inputSize; k++)
                {
                    activation += trainingInput[k] * LayerWeights[k][j];
                }

                LayerData[j] = Utils.lrelu(activation);
            }
        }

        public float[] BackwardPassCPU(float[] trainingOutput)
        {
            for (int j = 0; j < layerSize; j++)
            {
                float error = (trainingOutput[j] - LayerData[j]);
                LayerError[j] = error * Utils.dlrelu(LayerData[j]);
            }

            return LayerError;
        }

        public float[] BackwardPassCPU(float[] trainingOutput, float[][] outputWeights)
        {
            for (int j = 0; j < layerSize; j++)
            {
                float error = 0.0f;

                for(int k = 0; k < trainingOutput.Length; k++)
                {
                    error += trainingOutput[k] * outputWeights[j][k];
                }

                LayerError[j] = error * Utils.dlrelu(LayerData[j]);
            }

            return LayerError;
        }

        public void BackPropogationCPU(float[] outputError, float[] inputLayerData, float learningWeight)
        {
            for (int j = 0; j < layerSize; j++)
            {
                LayerBias[j] += outputError[j] * learningWeight;
                for (int k = 0; k < inputSize; k++)
                {
                    LayerWeights[k][j] += inputLayerData[k] * outputError[j] * learningWeight;
                }
            }
        }

        public void Dispose()
        {
            if(dLayerWeights != null) 
            {
                dLayerWeights.Dispose();
            }

            if (dLayerData != null)
            {
                dLayerData.Dispose();
            }

            if (dLayerBias != null)
            {
                dLayerBias.Dispose();
            }

            if (dLayerError != null)
            {
                dLayerError.Dispose();
            }
        }
    }
}