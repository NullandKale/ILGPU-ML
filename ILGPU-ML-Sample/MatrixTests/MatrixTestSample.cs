using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU_ML.Math;

namespace ILGPU_ML.MatrixTests
{
    public static class MatrixTestSample
    {
        private static Context context;
        private static Accelerator device;

        public static void Run()
        {
            bool debug = false;
            context = Context.Create(builder => builder.CPU().Cuda().EnableAlgorithms().Optimize(debug ? OptimizationLevel.Debug : OptimizationLevel.O1));
            device = context.GetPreferredDevice(preferCPU: false)
                                      .CreateAccelerator(context);

            MatrixMath math = new MatrixMath(context, device, 0.95f);

            Console.ReadLine();
        }
    }
}
