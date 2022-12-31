using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU_ML.VirtualMemory;
using ILGPU_ML.Mnist;
using ILGPU_ML.MatrixTests;
using System.Diagnostics;

namespace ILGPU_ML
{
    internal class Program
    {
        static void Main(string[] args)
        {
            AllocationTest.Run();
            DeallocationTest.Run();
            //CompleteTest.Run();

            MatrixTestSample.Run();

            //XorNetwork.Run();
            //MnistNetwork.Run();
        }
    }
}