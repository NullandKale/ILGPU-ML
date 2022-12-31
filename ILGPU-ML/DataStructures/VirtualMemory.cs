using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU_ML.DataStructures;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;

namespace ILGPU_ML.DataStructures
{
    public class Stride
    {
        public const int DenseX = 0;
        public const int DenseY = 1;
    }

    public struct Allocation
    {
        public long id;
        public long size;

        public Allocation(long id, long size)
        {
            this.id = id;
            this.size = size;
        }

        public override string? ToString()
        {
            return $"[{id}, {size}]";
        }
    }

    public class VirtualMemory<T> : IDisposable where T : unmanaged
    {
        private Accelerator device;
        private MemoryBuffer1D<T, Stride1D.Dense> backingMemoryBuffer;
        private MemoryBuffer1D<long, Stride1D.Dense> pointersBuffer;
        private List<long> pointers;
        private long sizeOfT;
        private long memorySize;
        private long currentLength;
        private List<Allocation> Allocations;
        private List<Allocation> Deallocations;
        private Comparer<Allocation> pointerCompare;

        public unsafe VirtualMemory(Accelerator device, float allocationPercent)
        {
            this.device = device;
            sizeOfT = sizeof(T);
            memorySize = (long)(device.MemorySize * allocationPercent) / sizeOfT;
            pointers = new List<long>();
            Allocations= new List<Allocation>();
            Deallocations = new List<Allocation>();
            backingMemoryBuffer = device.Allocate1D<T>(memorySize);
            currentLength = 0;

            pointerCompare = Comparer<Allocation>.Create((a, b) =>
            {
                long aPointer = GetPointer(a.id);
                long bPointer = GetPointer(b.id);
                return aPointer.CompareTo(bPointer);
            });
        }

        public unsafe VirtualMemory(Accelerator device, long bytesToAllocate)
        {
            this.device = device;
            sizeOfT = sizeof(T);
            memorySize = bytesToAllocate / sizeOfT;
            pointers = new List<long>();
            Allocations = new List<Allocation>();
            Deallocations = new List<Allocation>();
            backingMemoryBuffer = device.Allocate1D<T>(memorySize);
            currentLength = 0;

            pointerCompare = Comparer<Allocation>.Create((a, b) =>
            {
                long aPointer = GetPointer(a.id);
                long bPointer = GetPointer(b.id);
                return aPointer.CompareTo(bPointer);
            });
        }

        private long GetNextPointerID()
        {
            for(int i = 0; i < pointers.Count; i++)
            {
                if (pointers[i] == -1)
                {
                    pointers[i] = currentLength;
                    return i;
                }
            }

            long id = pointers.Count;
            pointers.Add(currentLength);

            return id;
        }

        private Allocation Allocate(long size)
        {
            if (currentLength + size < memorySize)
            {
                long id = GetNextPointerID();
                currentLength += size;

                Allocation a = new Allocation(id, size);

                Allocations.Add(a);
                return a;
            }
            else
            {
                throw new ArgumentOutOfRangeException("Not enough memory allocated");
            }
        }

        public HVirtualAllocation1D<T> Allocate1D(long size)
        {
            lock(this)
            {
                Allocation a = Allocate(size);
                VirtualAllocation1D<T> allocation = new VirtualAllocation1D<T>(a.size, a.id);
                return new HVirtualAllocation1D<T>(this, allocation);
            }
        }

        public void Deallocate(VirtualAllocation1D<T> allocation)
        {
            lock(this)
            {
                Allocation toRemove = new Allocation(allocation.id, allocation.size);
                RemoveAllocation(toRemove);
                Deallocations.Add(toRemove);
            }
        }

        public dVirtualMemory<T> GetD()
        {
            lock(this)
            {
                if(pointersBuffer == null)
                {
                    pointersBuffer = device.Allocate1D(pointers.ToArray());
                }
                else if (pointersBuffer.Length != pointers.Count)
                {
                    pointersBuffer.Dispose();
                    pointersBuffer = device.Allocate1D(pointers.ToArray());
                }
                else
                {
                    pointersBuffer.CopyFromCPU(pointers.ToArray());
                }
                
                return new dVirtualMemory<T>(backingMemoryBuffer, pointersBuffer, sizeOfT, memorySize, currentLength);
            }
        }

        public double GetPrecentUsed()
        {
            return ((double)currentLength / (double)memorySize * 100.0);
        }

        public void PrintStats()
        {
            string toPrint = "";

            lock(this)
            {
                double percentageUsed = GetPrecentUsed();
                toPrint = $"Virtual Memory Type: {typeof(T).Name}\n";
                toPrint += $"{percentageUsed.ToString("0.00")}% utilized\n";
                toPrint += $"{Utils.FormatBytes(currentLength * sizeOfT)} of {Utils.FormatBytes(memorySize * sizeOfT)}\n";
                toPrint += $"{Allocations.Count} Allocations | {Deallocations.Count} waiting for reclaimation";
            }

            Console.WriteLine(toPrint);
        }

        public long GetPointer(long id)
        {
            lock(this)
            {
                return pointers[(int)id];// cast might be an issue
            }
        }

        private void SetPointer(long id, long val)
        {
            pointers[(int)id] = val;
        }

        private void RemoveAllocation(Allocation toRemove)
        {
            List<Allocation> allocations = new List<Allocation>(Allocations.Count - 1);

            for(int i = 0; i < Allocations.Count; i++)
            {
                Allocation a = Allocations[i];
                if(!(a.id == toRemove.id && a.size == toRemove.size))
                {
                    allocations.Add(a);
                }
            }

            Allocations = allocations;
        }

        private Allocation GetAllocationAfterDislocation(Allocation dislocation)
        {
            long EndOfDislocation = GetPointer(dislocation.id) + dislocation.size;

            for(int i = 0; i < Allocations.Count; i++)
            {
                Allocation a = Allocations[i];
                long aPointer = GetPointer(a.id);

                if(aPointer == EndOfDislocation)
                {
                    return a;
                }
            }

            throw new Exception("Weird Allocation shape");
        }

        private void FindNewCurrentLength()
        {
            if(Allocations.Count == 0)
            {
                currentLength = 0;
                return;
            }

            SortedSet<Allocation> allocations = new SortedSet<Allocation>(Allocations, pointerCompare);
            Allocation last = allocations.Last();
            currentLength = GetPointer(last.id) + last.size;
        }

        private void RemoveDislocationsAfterCurrentLength()
        {
            List<Allocation> dislocations = new List<Allocation>();

            for (int i = 0; i < Deallocations.Count; i++)
            {
                Allocation a = Deallocations[i];
                long pointer = GetPointer(a.id);

                if (pointer < currentLength)
                {
                    dislocations.Add(a);
                }
                else
                {
                    SetPointer(a.id, -1);
                }
            }

            Deallocations = dislocations;
        }


        private void MergeDislocations()
        {
            if(Deallocations.Count == 0)
            {
                return;
            }

            List<Allocation> dislocations = new List<Allocation>(new SortedSet<Allocation>(Deallocations, pointerCompare));

            List<Allocation> compressedDislocations = new List<Allocation>();

            Allocation current = dislocations[0];

            for (int i = 1; i < dislocations.Count; i++)
            {
                Allocation next = dislocations[i];
                long currentEnd = GetPointer(current.id) + current.size;
                long nextPointer = GetPointer(next.id);

                // if the current ends at the point next begins just add the size to the current
                if (currentEnd == nextPointer)
                {
                    current.size += next.size;
                    SetPointer(next.id, -1);
                }
                else // current and next are discontinuous
                {
                    compressedDislocations.Add(current);
                    current = next;
                }
            }

            compressedDislocations.Add(current);

            Deallocations = compressedDislocations;
        }

        private void Move(Allocation freeSpace, Allocation toMove)
        {
            long freeSpaceBeginning = GetPointer(freeSpace.id);
            long toMoveBeginning = GetPointer(toMove.id);

            long totalMoved = 0;

            // while there is still more of toMove to ... move
            while(totalMoved < toMove.size)
            {
                // calculate amount of toMove left in the old location
                long toMoveLeft = toMove.size - totalMoved;

                // get the length of moveable space
                long length = XMath.Min(freeSpace.size, toMoveLeft);

                // move the moveable space from the location of 
                ArrayView1D<T, Stride1D.Dense> slice = backingMemoryBuffer.View.SubView(toMoveBeginning + totalMoved, length);
                ArrayView1D<T, Stride1D.Dense> dest = backingMemoryBuffer.View.SubView(freeSpaceBeginning + totalMoved, length);
                dest.CopyFrom(slice);    
                
                totalMoved += length;
            }

            // update freespace pointer to the new space it is in
            SetPointer(freeSpace.id, freeSpaceBeginning + totalMoved);

            // update toMove pointer to the new space it is in
            SetPointer(toMove.id, freeSpaceBeginning);
        }

        private void CleanupPointers()
        {
            // find the last valid pointer

            int lastPointer = 0;

            for(int i = 0; i < pointers.Count; i++)
            {
                if (pointers[i] != -1)
                {
                    lastPointer = i;
                }
            }

            // remove all unused pointers afterwards
            lastPointer++;

            if(lastPointer < pointers.Count)
            {
                pointers.RemoveRange(lastPointer, pointers.Count - lastPointer);
            }
        }

        private void Cleanup()
        {
            // removes all dislocations after the last allocation
            FindNewCurrentLength();
            RemoveDislocationsAfterCurrentLength();

            // merges all dislocations
            MergeDislocations();
        }

        public void Compact()
        {
            Cleanup();

            if(Deallocations.Count == 0)
            {
                CleanupPointers();
                return;
            }

            List<Allocation> dislocations = new List<Allocation>(new SortedSet<Allocation>(Deallocations, pointerCompare));

            // get first dislocation
            Allocation d = dislocations[0];

            // for each dislocation
            for (int i = 0; i < dislocations.Count; i++)
            {
                // if it is not the last dislocation
                if(i != dislocations.Count - 1)
                {
                    // while the dislocations are discontinuous
                    while (GetPointer(d.id) + d.size != GetPointer(dislocations[i + 1].id))
                    {
                        // get the next allocation after this dislocation
                        Allocation toMove = GetAllocationAfterDislocation(d);

                        // slide allocation up into dislocation
                        // and update dislocation pointer
                        Move(d, toMove);
                    }

                    // once the dislocation is continuous
                    // merge dislocation
                    d.size += dislocations[i + 1].size;
                }
                else
                {
                    // while the last dislocation does not hit the end of the allocations
                    while (GetPointer(d.id) + d.size != currentLength)
                    {
                        // get the next allocation after this dislocation
                        Allocation toMove = GetAllocationAfterDislocation(d);

                        // slide allocation up into dislocation
                        // and update dislocation pointer
                        Move(d, toMove);
                    }

                }
            }

            currentLength -= d.size;

            // remove deallocations from pointers
            for(int i = 0; i < Deallocations.Count; i++)
            {
                SetPointer(Deallocations[i].id, -1);
            }

            CleanupPointers();

            Deallocations.Clear();
        }

        public void Dispose()
        {
            backingMemoryBuffer.Dispose();

            if(pointersBuffer != null)
            {
                pointersBuffer.Dispose();
            }
        }
    }

    public struct dVirtualMemory<T> where T : unmanaged
    {
        public ArrayView1D<T, Stride1D.Dense> backingMemoryBuffer;
        public ArrayView1D<long, Stride1D.Dense> pointers;
        public long sizeOfT;
        public long memorySize;
        public long currentLength;

        public dVirtualMemory(MemoryBuffer1D<T, Stride1D.Dense> backingMemoryBuffer, MemoryBuffer1D<long, Stride1D.Dense> pointers, long sizeOfT, long memorySize, long currentLength)
        {
            this.backingMemoryBuffer = backingMemoryBuffer;
            this.pointers = pointers;
            this.sizeOfT = sizeOfT;
            this.memorySize = memorySize;
            this.currentLength = currentLength;
        }

        public long GetPointer(long id)
        {
            return pointers[id];
        }
    }

    public class HVirtualAllocation1D<T> : IDisposable where T : unmanaged
    {
        private bool isDisposed;
        private VirtualMemory<T> backing;
        private VirtualAllocation1D<T> data;

        public HVirtualAllocation1D(VirtualMemory<T> backing, VirtualAllocation1D<T> data)
        {
            isDisposed = false;
            this.backing = backing;
            this.data = data;
        }

        public void Dispose()
        {
            isDisposed = true;
            backing.Deallocate(data);
        }

        public VirtualAllocation1D<T> Get()
        {
            if (!isDisposed)
            {
                return data;
            }
            else
            {
                throw new InvalidOperationException("Allocation Disposed");
            }
        }
    }

    public struct VirtualAllocation1D<T> where T : unmanaged
    {
        public long id;
        public long size;

        public VirtualAllocation1D(long size, long id)
        {
            this.size = size;
            this.id = id;
        }

        public ArrayView1D<T, Stride1D.Dense> GetArrayView(dVirtualMemory<T> allocator)
        {
            return allocator.backingMemoryBuffer.SubView(allocator.GetPointer(id), size);
        }

        public T Get(dVirtualMemory<T> allocator, long index)
        {
            return allocator.backingMemoryBuffer[allocator.GetPointer(id) + index];
        }

        public void Set(dVirtualMemory<T> allocator, long index, T val)
        {
            allocator.backingMemoryBuffer[allocator.GetPointer(id) + index] = val;
        }
    }
}
