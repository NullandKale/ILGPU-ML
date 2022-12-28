using ILGPU_ML.DataStructures;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ILGPU_ML.Math
{
    public struct Vector<T> where T : unmanaged
    {
        public VirtualAllocation1D<T> data;

        public Vector(VirtualMemory<T> memory, int size)
        {
            data = memory.Allocate1D(size);
        }

        public T Get(dVirtualMemory<T> memory, int index)
        {
            return data.Get(memory, index);
        }

        public void Set(dVirtualMemory<T> memory, int index, T val)
        {
            data.Set(memory, index, val);
        }
    }
}
