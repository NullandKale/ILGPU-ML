using ILGPU.Runtime;
using ILGPU;
using ILGPU_ML.DataStructures;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Algorithms;
using System.Diagnostics;

namespace ILGPU_ML.VirtualMemory
{
    public class CompleteTest : IDisposable
    {
        public static void Run()
        {
            using CompleteTest test = new CompleteTest();

            int iterations = 20;

            for(int i = 0; i < iterations; i++)
            {
                Stopwatch timer = Stopwatch.StartNew();
                test.Test();
                timer.Stop();
                Console.WriteLine($"test {i} done in {timer.Elapsed.TotalSeconds}");
                Console.WriteLine($"Estimated time left: {timer.Elapsed * (iterations - i)}");
            }
        }

        private Context context;
        private Accelerator device;
        private VirtualMemory<long> memory;
        private Random rng;

        private List<HVirtualAllocation<long>> allocations;

        private Action<Index1D, dVirtualMemory<long>, VirtualAllocation<long>, long> Set1DKernel;
        private Action<Index1D, dVirtualMemory<long>, VirtualAllocation<long>, ArrayView1D<long, Stride1D.Dense>, long> Check1DKernel;

        public CompleteTest()
        {
            bool debug = false;
            context = Context.Create(builder => builder.CPU().Cuda().EnableAlgorithms().Optimize(debug ? OptimizationLevel.Debug : OptimizationLevel.O1));
            device = context.GetPreferredDevice(preferCPU: debug)
                                      .CreateAccelerator(context);

            rng = new Random(1);
            allocations = new List<HVirtualAllocation<long>>();

            memory = new VirtualMemory<long>(device, 0.95f);

            Set1DKernel = device.LoadAutoGroupedStreamKernel<Index1D, dVirtualMemory<long>, VirtualAllocation<long>, long>(SetIndex1D);
            Check1DKernel = device.LoadAutoGroupedStreamKernel<Index1D, dVirtualMemory<long>, VirtualAllocation<long>, ArrayView1D<long, Stride1D.Dense>, long>(CheckIndex1D);
        }

        public void Test(int percent)
        {
            FillRandom(percent);
            TestAllAllocations();

            SetAll();

            RemoveRandom((int)(allocations.Count * 0.1f));
            memory.Compact();

            CheckAll();

            RemoveRandom((int)(allocations.Count * 0.1f));

            FillRandom(percent * 0.1f);

            memory.Compact();

            TestAllAllocations();

            FillRandom(percent * 0.1f);

            int tenthOfAllocations = (int)(allocations.Count * 0.1f);

            for (int i = 0; i < 10; i++)
            {
                RemoveRandom(tenthOfAllocations);
                memory.Compact();

                TestAllAllocations();

                FillRandom(percent * 0.1f);
            }

            RemoveRandom(allocations.Count);
            memory.Compact();

            if (memory.GetPrecentUsed() != 0)
            {
                throw new Exception("allocator should have zero allocations");
            }
        }

        public void Test()
        {
            Test(10);
            Test(20);
            Test(30);
            Test(40);
            Test(50);
            Test(60);
            Test(70);
            Test(80);
            Test(90);
            Test(100);
        }

        public bool TryAllocateRandom()
        {
            try
            {
                //                                                                 1 b to 100MB
                HVirtualAllocation<long> allocation = memory.Allocate1D(rng.Next(1, 1024 * 1024 * 100 / 8));
                allocations.Add(allocation);
                return true;
            }
            catch (Exception e)
            {
                return false;
            }
        }

        public void FillRandom(float fillPercent)
        {
            while(TryAllocateRandom() && memory.GetPrecentUsed() < fillPercent)
            {

            }
        }

        public void RemoveRandom(int count = 1)
        {
            int toRemove = XMath.Min(count, allocations.Count);
            for(int i = 0; i < toRemove; i++)
            {
                int randomSelection = rng.Next(allocations.Count);
                var selection = allocations[randomSelection];
                allocations.RemoveAt(randomSelection);
                selection.Dispose();
            }
        }

        public bool TestAllocation(HVirtualAllocation<long> allocation)
        {
            Set(allocation);
            return Check(allocation);
        }

        public void SetAll()
        {
            for (int i = 0; i < allocations.Count; i++)
            {
                Set(allocations[i]);
            }
        }

        public void CheckAll()
        {
            for (int i = 0; i < allocations.Count; i++)
            {
                if (!Check(allocations[i]))
                {
                    throw new Exception("Check Failed!");
                }
            }
        }

        public void TestAllAllocations()
        {
            SetAll();
            CheckAll();
            
            for(int i = 0; i < allocations.Count; i++)
            {
                if(!TestAllocation(allocations[i]))
                {
                    throw new Exception("Check Failed!");
                }
            }
        }

        private static void SetIndex1D(Index1D index, dVirtualMemory<long> memory, VirtualAllocation<long> data, long offset)
        {
            data.Set(memory, index, data.id);
        }

        private static void CheckIndex1D(Index1D index, dVirtualMemory<long> memory, VirtualAllocation<long> data, ArrayView1D<long, Stride1D.Dense> invalidCount, long offset)
        {
            if (data.Get(memory, index) != data.id)
            {
                Atomic.Exchange(ref invalidCount[0], data.id + 1);
            }
        }

        private void Set(HVirtualAllocation<long> allocation)
        {
            Set1DKernel((int)allocation.Get().size, memory.GetD(), allocation.Get(), memory.GetPointer(allocation.Get().id));
            device.Synchronize();
        }

        private bool Check(HVirtualAllocation<long> allocation)
        {
            using MemoryBuffer1D<long, Stride1D.Dense> error = device.Allocate1D<long>(new long[] { 0L });

            Check1DKernel((int)allocation.Get().size, memory.GetD(), allocation.Get(), error, memory.GetPointer(allocation.Get().id));
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

        public void Dispose()
        {
            memory.Dispose();
            device.Dispose();
            context.Dispose();
        }

    }
}
