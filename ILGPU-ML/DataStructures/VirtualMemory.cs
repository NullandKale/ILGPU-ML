using ILGPU;
using ILGPU.Runtime;
using ILGPU_ML.DataStructures;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ILGPU_ML.DataStructures
{
    public class Stride
    {
        public const int DenseX = 0;
        public const int DenseY = 1;
    }

    public class VirtualMemory<T> : IDisposable where T : unmanaged
    {
        private Accelerator device;
        private MemoryBuffer1D<T, Stride1D.Dense> backingMemoryBuffer;
        private long sizeOfT;
        private long memorySize;
        private long currentLength;

        public unsafe VirtualMemory(Accelerator device, float allocationPercent)
        {
            this.device = device;
            sizeOfT = sizeof(T);
            memorySize = (long)(device.MemorySize * allocationPercent) / sizeOfT;
            backingMemoryBuffer = device.Allocate1D<T>(memorySize);
            currentLength = 0;
        }

        public unsafe VirtualMemory(Accelerator device, long bytesToAllocate)
        {
            this.device = device;
            sizeOfT = sizeof(T);
            memorySize = bytesToAllocate / sizeOfT;
            backingMemoryBuffer = device.Allocate1D<T>(memorySize);
            currentLength = 0;
        }

        public VirtualAllocation1D<T> Allocate1D(int size)
        {
            lock(this)
            {
                if(currentLength + size < memorySize)
                {
                    VirtualAllocation1D<T> allocation = new VirtualAllocation1D<T>(size, size * sizeOfT, currentLength);
                    currentLength += size;
                    return allocation;
                }
                else
                {
                    throw new ArgumentOutOfRangeException("Not enough memory allocated");
                }
            }
        }

        public VirtualAllocation2D<T> Allocate2D(Vec2i size)
        {
            lock (this)
            {
                if (currentLength + size.GetLength() < memorySize)
                {
                    VirtualAllocation2D<T> allocation = new VirtualAllocation2D<T>(size, size.GetLength() * sizeOfT, currentLength);
                    currentLength += size.GetLength();
                    return allocation;
                }
                else
                {
                    throw new ArgumentOutOfRangeException("Not enough memory allocated");
                }
            }
        }

        public dVirtualMemory<T> GetD()
        {
            lock(this)
            {
                return new dVirtualMemory<T>(backingMemoryBuffer, sizeOfT, memorySize, currentLength);
            }
        }

        public void PrintStats()
        {
            string toPrint = "";

            lock(this)
            {
                double percentageUsed = ((double)currentLength / (double)memorySize * 100.0);
                toPrint = $"Virtual Memory <{typeof(T).Name}> {percentageUsed.ToString("0.00")}% used {Utils.FormatBytes(currentLength * sizeOfT)} : {Utils.FormatBytes(memorySize * sizeOfT)}";
            }

            Console.WriteLine(toPrint);
        }

        public void Dispose()
        {
            backingMemoryBuffer.Dispose();
        }
    }

    public struct dVirtualMemory<T> where T : unmanaged
    {
        public ArrayView1D<T, Stride1D.Dense> backingMemoryBuffer;
        public long sizeOfT;
        public long memorySize;
        public long currentLength;

        public dVirtualMemory(MemoryBuffer1D<T, Stride1D.Dense> backingMemoryBuffer, long sizeOfT, long memorySize, long currentLength)
        {
            this.backingMemoryBuffer = backingMemoryBuffer;
            this.sizeOfT = sizeOfT;
            this.memorySize = memorySize;
            this.currentLength = currentLength;
        }
    }

    public struct VirtualAllocation1D<T> where T : unmanaged
    {
        public long length;
        public long pointer;
        public long size;

        public VirtualAllocation1D(long size, long length, long pointer)
        {
            this.length = length;
            this.size = size;
            this.pointer = pointer;
        }

        public ArrayView1D<T, Stride1D.Dense> GetArrayView(dVirtualMemory<T> allocator)
        {
            return allocator.backingMemoryBuffer.SubView(pointer, size);
        }

        public T Get(dVirtualMemory<T> allocator, long index)
        {
            return allocator.backingMemoryBuffer[pointer + index];
        }

        public void Set(dVirtualMemory<T> allocator, long index, T val)
        {
            allocator.backingMemoryBuffer[pointer + index] = val;
        }
    }

    public struct VirtualAllocation2D<T> where T : unmanaged
    {
        public Vec2i size;
        public long length;
        public long pointer;
        public int Density = Stride.DenseY;

        public VirtualAllocation2D(Vec2i size, long length, long pointer)
        {
            this.length = length;
            this.size = size;
            this.pointer = pointer;
        }

        public ArrayView2D<T, Stride2D.DenseX> GetArrayViewDenseX(dVirtualMemory<T> allocator)
        {
            return allocator.backingMemoryBuffer.SubView(pointer, size.GetLength()).As2DDenseXView(size);
        }

        public ArrayView2D<T, Stride2D.DenseY> GetArrayViewDenseY(dVirtualMemory<T> allocator)
        {
            return allocator.backingMemoryBuffer.SubView(pointer, size.GetLength()).As2DDenseYView(size);
        }

        public long GetIndex(long x, long y)
        {
            long index = 0;

            if (Density == Stride.DenseY)
            {
                index = x * size.x + y;
            }
            else if (Density == Stride.DenseX)
            {
                index = y * size.y + x;
            }

            return index;
        }

        public T Get(dVirtualMemory<T> allocator, long x, long y)
        {
            return allocator.backingMemoryBuffer[pointer + GetIndex(x, y)];
        }

        public void Set(dVirtualMemory<T> allocator, long x, long y, T val)
        {
            allocator.backingMemoryBuffer[pointer + GetIndex(x, y)] = val;
        }
    }
}
