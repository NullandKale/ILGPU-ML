using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Algorithms;
using ILGPU_ML.DataStructures;

namespace ILGPU_ML.VirtualMemory
{
    public static class AllocationTest
    {
        private static void SetIndex1D(Index1D index, dVirtualMemory<long> memory, VirtualAllocation<long> data, long offset)
        {
            data.Set(memory, index, index + offset);
        }

        private static void CheckIndex1D(Index1D index, dVirtualMemory<long> memory, VirtualAllocation<long> data, ArrayView1D<long, Stride1D.Dense> invalidCount, long offset)
        {
            if(data.Get(memory, index) != index + offset)
            {
                Atomic.Exchange(ref invalidCount[0], index.X);
            }
        }

        private static Action<Index1D, dVirtualMemory<long>, VirtualAllocation<long>, long> Set1DKernel;
        private static Action<Index1D, dVirtualMemory<long>, VirtualAllocation<long>, ArrayView1D<long, Stride1D.Dense>, long> Check1DKernel;

        private static void Set(Accelerator device, VirtualMemory<long> memory, VirtualAllocation<long> allocation)
        {
            Set1DKernel((int)allocation.size, memory.GetD(), allocation, memory.GetPointer(allocation.id));
            device.Synchronize();
        }

        private static bool Check(Accelerator device, VirtualMemory<long> memory, VirtualAllocation<long> allocation)
        {
            MemoryBuffer1D<long, Stride1D.Dense> error = device.Allocate1D<long>(new long[]{ 0L});

            Check1DKernel((int)allocation.size, memory.GetD(), allocation, error, memory.GetPointer(allocation.id));
            device.Synchronize();

            long errorCount = error.GetAsArray1D()[0];

            if(errorCount > 0)
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        public static void Run()
        {
            bool debug = false;
            using Context context = Context.Create(builder => builder.CPU().Cuda().EnableAlgorithms().Optimize(debug ? OptimizationLevel.Debug : OptimizationLevel.O1));
            using Accelerator device = context.GetPreferredDevice(preferCPU: debug)
                                      .CreateAccelerator(context);

            using VirtualMemory<long> memory = new VirtualMemory<long>(device, 0.95f);

            Set1DKernel = device.LoadAutoGroupedStreamKernel<Index1D, dVirtualMemory<long>, VirtualAllocation<long>, long>(SetIndex1D);
            Check1DKernel = device.LoadAutoGroupedStreamKernel<Index1D, dVirtualMemory<long>, VirtualAllocation<long>, ArrayView1D<long, Stride1D.Dense>, long>(CheckIndex1D);

            List<HVirtualAllocation<long>> allocations1D = new List<HVirtualAllocation<long>>();

            Random rng = new Random();

            for(int i = 0; i < 1000; i++)
            {
                try
                {
                    allocations1D.Add(memory.Allocate1D(rng.Next(10, 1024 * 1024 * 10)));
                }
                catch(Exception e) 
                {
                    //Console.WriteLine(e.ToString());
                    break;
                }
            }

            memory.PrintStats();

            for(int i = 0; i < 10; i++)
            {

            }

            Console.WriteLine($"Checking {allocations1D.Count} Allocations1D");

            for (int i = 0; i < allocations1D.Count; i++)
            {
                Set(device, memory, allocations1D[i].Get());
            }

            for (int i = 0; i < allocations1D.Count; i++)
            {
                if(!Check(device, memory, allocations1D[i].Get()))
                {
                    Console.WriteLine($"Check 1D {i} Failed");
                }
            }

            for (int i = 0; i < allocations1D.Count; i++)
            {
                Set(device, memory, allocations1D[i].Get());
                if (!Check(device, memory, allocations1D[i].Get()))
                {
                    Console.WriteLine($"Check 1D {i} Failed");
                }
            }
        }
    }
}
