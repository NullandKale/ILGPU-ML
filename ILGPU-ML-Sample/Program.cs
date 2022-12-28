using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU_ML.VirtualMemory;
using ILGPU_ML.Mnist;
using ILGPU_ML.MatrixTests;

namespace ILGPU_ML
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //XorNetwork.Run();
            //MnistNetwork.Run();
            while (true)
            {
                VMSample.Run();

            }

            //MatrixTestSample.Run();
        }
    }
}