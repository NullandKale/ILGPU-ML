using ILGPU_ML.DataStructures;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ILGPU_ML.Math
{
    public class HMatrix<T> : IDisposable where T : unmanaged
    {
        private HVirtualAllocation1D<T> allocation;
        private Matrix<T> data;

        public HMatrix(VirtualMemory<T> memory, Vec2i size)
        {
            allocation = memory.Allocate1D(size.GetLength());
            data = new Matrix<T>(allocation.Get(), size);
        }

        public Matrix<T> Get()
        {
            return data;
        }

        public void Dispose()
        {
            allocation.Dispose();
        }
    }

    public struct Matrix<T> where T : unmanaged
    {
        public VirtualAllocation1D<T> data;
        public int Density = Stride.DenseX;
        public Vec2i size;

        public Matrix(VirtualAllocation1D<T> data, Vec2i size)
        {
            this.size = size;
            this.data = data;
        }

        public long GetIndex(long x, long y)
        {
            long index = 0;

            if (Density == Stride.DenseY)
            {
                index = x * size.y + y;
            }
            else if (Density == Stride.DenseX)
            {
                index = y * size.x + x;
            }

            return index;
        }

        public T Get(dVirtualMemory<T> memory, Vec2i index)
        {
            long i = GetIndex(index.x, index.y);
            return data.Get(memory, i);
        }

        public T Get(dVirtualMemory<T> memory, int x, int y)
        {
            long i = GetIndex(x, y);
            return data.Get(memory, i);
        }

        public T GetTransposed(dVirtualMemory<T> memory, Vec2i index)
        {
            long i = GetIndex(index.y, index.x);
            return data.Get(memory, i);
        }

        public T GetTransposed(dVirtualMemory<T> memory, int x, int y)
        {
            long i = GetIndex(y, x);
            return data.Get(memory, i);
        }

        public void Set(dVirtualMemory<T> memory, Vec2i index, T val)
        {
            long i = GetIndex(index.x, index.y);
            data.Set(memory, i, val);
        }

        public void Set(dVirtualMemory<T> memory, int x, int y, T val)
        {
            long i = GetIndex(x, y);
            data.Set(memory, i, val);
        }

        public void SetTransposed(dVirtualMemory<T> memory, Vec2i index, T val)
        {
            long i = GetIndex(index.y, index.x);
            data.Set(memory, i, val);
        }

        public void SetTransposed(dVirtualMemory<T> memory, int x, int y, T val)
        {
            long i = GetIndex(y, x);
            data.Set(memory, i, val);
        }
    }
}
