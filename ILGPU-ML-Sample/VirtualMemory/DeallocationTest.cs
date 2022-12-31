using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU_ML.DataStructures;

namespace ILGPU_ML.VirtualMemory
{
    public static class DeallocationTest
    {
        public static void Run()
        {
            bool debug = false;
            using Context context = Context.Create(builder => builder.CPU().Cuda().EnableAlgorithms().Optimize(debug ? OptimizationLevel.Debug : OptimizationLevel.O1));
            using Accelerator device = context.GetPreferredDevice(preferCPU: debug)
                                      .CreateAccelerator(context);

            using VirtualMemory<long> memory = new VirtualMemory<long>(device, 200L);

            var a1 = memory.Allocate1D(3); // 0, 3
            
            var d1 = memory.Allocate1D(2); // 3, 2

            var a2 = memory.Allocate1D(3); // 5, 3
            var a3 = memory.Allocate1D(3); // 8, 3

            var d2 = memory.Allocate1D(2); // 11, 2
            var d3 = memory.Allocate1D(2); // 13, 2

            var a4 = memory.Allocate1D(3); // 15, 3

            var d4 = memory.Allocate1D(4); // 18, 4

            memory.PrintStats();

            d1.Dispose();
            d2.Dispose();
            d3.Dispose();
            d4.Dispose();

            memory.PrintStats();

            memory.Compact();

            memory.PrintStats();
        }
    }
}
