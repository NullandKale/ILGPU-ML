using ILGPU_ML.DataStructures;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ILGPU_ML.Math
{
    public struct Matrix<T> where T : unmanaged
    {
        public VirtualAllocation2D<T> data;

        public Matrix(VirtualMemory<T> memory, Vec2i size)
        {
            data = memory.Allocate2D(size);
        }

        public T Get(dVirtualMemory<T> memory, Vec2i index)
        {
            return data.Get(memory, index.x, index.y);
        }

        public T Get(dVirtualMemory<T> memory, int x, int y)
        {
            return data.Get(memory, x, y);
        }

        public T GetTransposed(dVirtualMemory<T> memory, Vec2i index)
        {
            return data.Get(memory, index.y, index.x);
        }

        public T GetTransposed(dVirtualMemory<T> memory, int x, int y)
        {
            return data.Get(memory, y, x);
        }

        public void Set(dVirtualMemory<T> memory, Vec2i index, T val)
        {
            data.Set(memory, index.x, index.y, val);
        }

        public void Set(dVirtualMemory<T> memory, int x, int y, T val)
        {
            data.Set(memory, x, y, val);
        }

        public void SetTransposed(dVirtualMemory<T> memory, Vec2i index, T val)
        {
            data.Set(memory, index.y, index.x, val);
        }

        public void SetTransposed(dVirtualMemory<T> memory, int x, int y, T val)
        {
            data.Set(memory, y, x, val);
        }
    }
}
