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
    public static class VMSample
    {
        private static void SetIndex1D(Index1D index, dVirtualMemory<long> memory, VirtualAllocation1D<long> data, long offset)
        {
            data.Set(memory, index, index + offset);
        }

        private static void SetIndex2D(Index2D index, dVirtualMemory<long> memory, VirtualAllocation2D<long> data, long offset)
        {
            data.Set(memory, index.X, index.Y, data.GetIndex(index.X, index.Y) + offset);
        }

        private static void CheckIndex1D(Index1D index, dVirtualMemory<long> memory, VirtualAllocation1D<long> data, ArrayView1D<long, Stride1D.Dense> invalidCount, long offset)
        {
            if(data.Get(memory, index) != index + offset)
            {
                Atomic.Add(ref invalidCount[0], 1);
            }
        }

        private static void CheckIndex2D(Index2D index, dVirtualMemory<long> memory, VirtualAllocation2D<long> data, ArrayView1D<long, Stride1D.Dense> invalidCount, long offset)
        {
            if (data.Get(memory, index.X, index.Y) != data.GetIndex(index.X, index.Y) + offset)
            {
                Atomic.Add(ref invalidCount[0], 1);
            }
        }

        private static Context context;
        private static Accelerator device;
        private static Action<Index1D, dVirtualMemory<long>, VirtualAllocation1D<long>, long> Set1DKernel;
        private static Action<Index2D, dVirtualMemory<long>, VirtualAllocation2D<long>, long> Set2DKernel;
        private static Action<Index1D, dVirtualMemory<long>, VirtualAllocation1D<long>, ArrayView1D<long, Stride1D.Dense>, long> Check1DKernel;
        private static Action<Index2D, dVirtualMemory<long>, VirtualAllocation2D<long>, ArrayView1D<long, Stride1D.Dense>, long> Check2DKernel;

        private static void Set(VirtualMemory<long> memory, VirtualAllocation1D<long> allocation)
        {
            Set1DKernel(allocation.size, memory.GetD(), allocation, allocation.pointer);
            device.Synchronize();
        }

        private static bool Check(VirtualMemory<long> memory, VirtualAllocation1D<long> allocation)
        {
            MemoryBuffer1D<long, Stride1D.Dense> error = device.Allocate1D<long>(1);

            Check1DKernel(allocation.size, memory.GetD(), allocation, error, allocation.pointer);
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

        private static void Set(VirtualMemory<long> memory, VirtualAllocation2D<long> allocation)
        {
            Set2DKernel(allocation.size, memory.GetD(), allocation, allocation.pointer);
            device.Synchronize();
        }

        private static bool Check(VirtualMemory<long> memory, VirtualAllocation2D<long> allocation)
        {
            MemoryBuffer1D<long, Stride1D.Dense> error = device.Allocate1D<long>(1);

            Check2DKernel(allocation.size, memory.GetD(), allocation, error, allocation.pointer);
            device.Synchronize();

            long errorCount = error.GetAsArray1D()[0];

            if (errorCount > 0)
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
            context = Context.Create(builder => builder.CPU().Cuda().EnableAlgorithms().Optimize(debug ? OptimizationLevel.Debug : OptimizationLevel.O1));
            device = context.GetPreferredDevice(preferCPU: true)
                                      .CreateAccelerator(context);

            using VirtualMemory<long> memory = new VirtualMemory<long>(device, 1024L * 1024L * 1024L * 16L);

            Set1DKernel = device.LoadAutoGroupedStreamKernel<Index1D, dVirtualMemory<long>, VirtualAllocation1D<long>, long>(SetIndex1D);
            Check1DKernel = device.LoadAutoGroupedStreamKernel<Index1D, dVirtualMemory<long>, VirtualAllocation1D<long>, ArrayView1D<long, Stride1D.Dense>, long>(CheckIndex1D);

            Set2DKernel = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<long>, VirtualAllocation2D<long>, long>(SetIndex2D);
            Check2DKernel = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<long>, VirtualAllocation2D<long>, ArrayView1D<long, Stride1D.Dense>, long>(CheckIndex2D);

            List<VirtualAllocation1D<long>> allocations1D = new List<VirtualAllocation1D<long>>();
            List<VirtualAllocation2D<long>> allocations2D = new List<VirtualAllocation2D<long>>();

            Random rng = new Random();

            for(int i = 0; i < 1000; i++)
            {
                try
                {
                    allocations1D.Add(memory.Allocate1D(rng.Next(10, 1024 * 1024 * 10)));
                }
                catch(Exception e) 
                {
                    Console.WriteLine(e.ToString());
                    break;
                }

                try
                {
                    allocations2D.Add(memory.Allocate2D(new Vec2i(rng.Next(10, 1024 * 10), rng.Next(10, 1024 * 10))));
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.ToString());
                    break;
                }
            }

            memory.PrintStats();

            Console.WriteLine($"Checking {allocations1D.Count} Allocations1D");

            for (int i = 0; i < allocations1D.Count; i++)
            {
                Set(memory, allocations1D[i]);
            }

            for (int i = 0; i < allocations1D.Count; i++)
            {
                if(!Check(memory, allocations1D[i]))
                {
                    Console.WriteLine($"Check 1D {i} Failed");
                }
            }

            for (int i = 0; i < allocations1D.Count; i++)
            {
                Set(memory, allocations1D[i]);
                if (!Check(memory, allocations1D[i]))
                {
                    Console.WriteLine($"Check 1D {i} Failed");
                }
            }

            Console.WriteLine($"Checking {allocations2D.Count} Allocations2D");

            for (int i = 0; i < allocations2D.Count; i++)
            {
                Set(memory, allocations2D[i]);
            }

            for (int i = 0; i < allocations2D.Count; i++)
            {
                if (!Check(memory, allocations2D[i]))
                {
                    Console.WriteLine($"Check 2D {i} Failed");
                }
            }

            for (int i = 0; i < allocations2D.Count; i++)
            {
                Set(memory, allocations2D[i]);
                if (!Check(memory, allocations2D[i]))
                {
                    Console.WriteLine($"Check 2D {i} Failed");
                }
            }


        }
    }
}
