using ILGPU_ML.DataStructures;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ILGPU_ML.Math
{
    public class HVector<T> : IDisposable where T : unmanaged
    {
        public HVirtualAllocation1D<T> allocation;
        public Vector<T> data;

        public HVector(VirtualMemory<T> memory, long size)
        {
            allocation = memory.Allocate1D(size);
            data = new Vector<T>(allocation.Get());
        }

        public void Dispose()
        {
            allocation.Dispose();
        }
    }

    public struct Vector<T> where T : unmanaged
    {
        public VirtualAllocation1D<T> data;

        public Vector(VirtualAllocation1D<T> data)
        {
            this.data = data;
        }

        public T Get(dVirtualMemory<T> memory, long index)
        {
            return data.Get(memory, index);
        }

        public void Set(dVirtualMemory<T> memory, long index, T val)
        {
            data.Set(memory, index, val);
        }
    }
}
