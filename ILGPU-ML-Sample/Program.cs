using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU_ML.VirtualMemory;
using ILGPU_ML.Mnist;
using ILGPU_ML.MatrixTests;
using System.Diagnostics;
using ILGPU_ML.Xor;

namespace ILGPU_ML
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //AllocationTest.Run();
            //DeallocationTest.Run();
            //CompleteTest.Run();

            MatrixTestSample.Run(int.MaxValue);

            //XorNetwork.Run();
            //MnistNetwork.Run();
        }
    }
}