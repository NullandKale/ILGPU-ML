using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU_ML.DataStructures;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO.Pipes;
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
        //private List<long> pointers;
        private long[] pointersArray;
        private int pointersCount;
        private long sizeOfT;
        private long memorySize;
        private long currentLength;
        private long currentCleanedUpLength;
        private List<Allocation> Allocations;
        private List<Allocation> Deallocations;
        private List<long> openIDs;
        private Comparer<Allocation> pointerCompare;

        private int maxSamplesToAverage = 1000;

        private int maxPointers = -1;

        private long allocationsSum;
        private Queue<(long allocationSize, DateTime timestamp)> lastNAllocations;

        private double compactionTimeSum;
        private int[] typeSums = new int[4];
        private Queue<(DateTime timestamp, TimeSpan duration, int type)> lastNCompactions;

        public unsafe VirtualMemory(Accelerator device, float allocationPercent)
        {
            this.device = device;
            sizeOfT = sizeof(T);
            memorySize = (long)(device.MemorySize * allocationPercent) / sizeOfT;
            pointersArray = new long[1];
            pointersCount = 0;
            Allocations = new List<Allocation>();
            Deallocations = new List<Allocation>();
            lastNAllocations = new Queue<(long allocationSize, DateTime timestamp)>();
            lastNCompactions = new Queue<(DateTime, TimeSpan, int)>();
            openIDs = new List<long>();
            backingMemoryBuffer = device.Allocate1D<T>(memorySize);
            currentLength = 0;
            currentCleanedUpLength = 0;

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
            pointersArray = new long[1];
            pointersCount = 0;
            openIDs = new List<long>();
            Allocations = new List<Allocation>();
            Deallocations = new List<Allocation>();
            backingMemoryBuffer = device.Allocate1D<T>(memorySize);
            lastNAllocations = new Queue<(long allocationSize, DateTime timestamp)>();
            lastNCompactions = new Queue<(DateTime, TimeSpan, int)>();
            currentLength = 0;
            currentCleanedUpLength = 0;

            pointerCompare = Comparer<Allocation>.Create((a, b) =>
            {
                long aPointer = GetPointer(a.id);
                long bPointer = GetPointer(b.id);
                return aPointer.CompareTo(bPointer);
            });
        }

        private long GetNextPointerID()
        {
            if(openIDs.Count > 0)
            {
                openIDs.Sort();
                long lowest = openIDs[0];
                openIDs.RemoveAt(0);
                if (pointersArray[(int)lowest] == -1)
                {
                    pointersArray[(int)lowest] = currentLength;
                    return lowest;
                }
                else
                {
                    throw new Exception("openID not open");
                }
            }

            long id = pointersCount;
            pointersCount++;
            if (id >= pointersArray.Length)
            {
                int newLength = pointersArray.Length * 2;
                Array.Resize(ref pointersArray, newLength);
            }

            pointersArray[id] = currentLength;

            if (id > maxPointers)
            {
                maxPointers = (int)id;
            }

            return id;
        }

        private void TraceAllocations(long size)
        {
            DateTime timestamp = DateTime.Now;
            allocationsSum += size;
            
            if(lastNAllocations.Count > maxSamplesToAverage)
            {
                var removed = lastNAllocations.Dequeue();
                allocationsSum -= removed.allocationSize;
            }

            lastNAllocations.Enqueue((size, timestamp));
        }

        private void TraceCompaction(TimeSpan duration, int compactionType)
        {
            DateTime timestamp = DateTime.Now;
            compactionTimeSum += duration.TotalSeconds;
            typeSums[compactionType]++;

            if (lastNCompactions.Count > 100)
            {
                var removed = lastNCompactions.Dequeue();
                compactionTimeSum -= removed.duration.TotalSeconds;
                typeSums[removed.type]--;
            }

            lastNCompactions.Enqueue((timestamp, duration, compactionType));
        }

        private (double compactionsPerSecond, double averageCompactionTime, string compactionTypes) GetCompactionsData()
        {
            if(lastNCompactions.Count < 1)
            {
                return (0, 0, "");
            }

            double cps = lastNCompactions.Count / (DateTime.Now - lastNCompactions.Peek().timestamp).TotalSeconds;

            string compactionTypeString = "";

            double totalCompations = lastNCompactions.Count;

            for(int i = 0; i < typeSums.Length; i++)
            {
                double avg = typeSums[i] / totalCompations;
                compactionTypeString += $"{avg.ToString("0.00")} ";
            }

            return (cps, compactionTimeSum / lastNCompactions.Count, compactionTypeString);
        }

        private double GetAllocationsData()
        {
            DateTime oldestTime = lastNAllocations.Peek().timestamp;
            return (double)allocationsSum / (DateTime.Now - oldestTime).TotalSeconds;
        }

        private Allocation Allocate(long size)
        {
            bool willFit = currentLength + size < memorySize;
            bool willFitWhenCompacted = currentCleanedUpLength + size < memorySize;
            var needsCompaction = NeedsCompaction();
            bool tryGC = !willFit && !willFitWhenCompacted;

            if(tryGC)
            {
                GC.Collect(2, GCCollectionMode.Aggressive, true, true);
                willFit = currentLength + size < memorySize;
                willFitWhenCompacted = currentCleanedUpLength + size < memorySize;
                needsCompaction = NeedsCompaction();
            }

            if(!willFit && willFitWhenCompacted)
            {
                Compact(3);
                needsCompaction = (false, -1);
                willFit = true;
            }

            if(needsCompaction.needsCompaction)
            {
                Compact(needsCompaction.type);
            }

            if (willFit)
            {
                long id = GetNextPointerID();
                currentLength += size;
                currentCleanedUpLength += size;

                Allocation a = new Allocation(id, size);

                Allocations.Add(a);
                TraceAllocations(size);

                return a;
            }
            else
            {
                throw new ArgumentOutOfRangeException("Not enough memory allocated");
            }
        }

        public HVirtualAllocation<T> Allocate1D(long size)
        {
            lock(this)
            {
                Allocation a = Allocate(size);
                VirtualAllocation<T> allocation = new VirtualAllocation<T>(a.size, a.id);
                return new HVirtualAllocation<T>(this, allocation);
            }
        }

        private (bool needsCompaction, int type) NeedsCompaction()
        {
            float dCount = Deallocations.Count;
            float aCount = Allocations.Count;

            float spaceToDeallocate = currentLength - currentCleanedUpLength;
            if(spaceToDeallocate >= currentLength && spaceToDeallocate != 0)
            {
                return (true, 1);
            }

            if(dCount > aCount * aCount)
            {
                return (true, 2);
            }

            return (false, -1);
        }

        public void Deallocate(VirtualAllocation<T> allocation)
        {
            lock(this)
            {
                Allocation toRemove = new Allocation(allocation.id, allocation.size);
                RemoveAllocation(toRemove);
                Deallocations.Add(toRemove);
                currentCleanedUpLength -= allocation.size;
            }
        }

        public dVirtualMemory<T> GetD()
        {
            lock(this)
            {
                if (pointersBuffer == null)
                {   
                    pointersBuffer = device.Allocate1D(pointersArray);
                }
                else if (pointersBuffer.Length != pointersArray.Length)
                {
                    pointersBuffer.Dispose();
                    pointersBuffer = device.Allocate1D(pointersArray);
                }
                else
                {
                    pointersBuffer.CopyFromCPU(pointersArray);
                }
                
                return new dVirtualMemory<T>(backingMemoryBuffer, pointersBuffer, sizeOfT, memorySize, currentLength);
            }
        }

        public double GetPrecentUsed()
        {
            return ((double)currentLength / (double)memorySize * 100.0);
        }

        public void PrintStats(string toAppend = "")
        {
            string toPrint = "";

            lock(this)
            {
                double percentageUsed = GetPrecentUsed();
                var compactionData = GetCompactionsData();
                toPrint = $"Virtual Memory Type: {typeof(T).Name}\n";
                toPrint += $"{percentageUsed.ToString("0.00")}% utilized\n";
                toPrint += $"{Utils.FormatBytes(currentLength * sizeOfT)} of {Utils.FormatBytes(memorySize * sizeOfT)}\n";
                toPrint += $"{Utils.FormatBytes(currentCleanedUpLength * sizeOfT)} of {Utils.FormatBytes(memorySize * sizeOfT)} if compacted\n";
                toPrint += $"{Utils.FormatBytes((long)GetAllocationsData())}/s allocated.\n";
                toPrint += $"{compactionData.compactionsPerSecond.ToString("0.00")} cps, {compactionData.averageCompactionTime.ToString("0.00")}s.\n";
                toPrint += $"{compactionData.compactionTypes}\n";
                toPrint += $"{maxPointers} max pointers\n";
                toPrint += $"{Allocations.Count} Allocations | {Deallocations.Count} waiting for reclaimation.";

            }

            Console.WriteLine(toPrint);
            if(toAppend != null && toAppend.Length > 0)
            {
                Console.WriteLine(toAppend);
            }
        }

        public long GetPointer(long id)
        {
            lock(this)
            {
                if(id < 0 || id > pointersCount)
                {
                    throw new Exception("Invalid GPU memory access");
                }

                return pointersArray[(int)id];// cast might be an issue
            }
        }

        private void SetPointer(long id, long val)
        {
            if (id < 0 || id > pointersCount)
            {
                throw new Exception("Invalid GPU memory access");
            }

            if(val == -1)
            {
                openIDs.Add((int)id);
            }

            pointersArray[(int)id] = val;
            //pointers[(int)id] = val;
        }

        private void RemoveAllocation(Allocation toRemove)
        {
            List<Allocation> allocations = new List<Allocation>(XMath.Max(Allocations.Count - 1, 1));

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

        private ArrayView1D<T, Stride1D.Dense> GetSlice(long pointer, long length)
        {
            return backingMemoryBuffer.View.SubView(pointer, length);
        }

        public ArrayView1D<T, Stride1D.Dense> GetSlice(HVirtualAllocation<T> allocation)
        {
            long pointer = GetPointer(allocation.Get().id);
            long length = allocation.Get().size;
            return GetSlice(pointer, length);
        }

        public ArrayView2D<T, Stride2D.DenseY> GetSlice(HVirtualAllocation<T> allocation, Vec2i size)
        {
            long pointer = GetPointer(allocation.Get().id);
            long length = allocation.Get().size;
            return GetSlice(pointer, length).As2DDenseYView(size);
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
                ArrayView1D<T, Stride1D.Dense> slice = GetSlice(toMoveBeginning + totalMoved, length);
                ArrayView1D<T, Stride1D.Dense> dest = GetSlice(freeSpaceBeginning + totalMoved, length);
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

            List<long> pointers = new List<long>(pointersArray);

            openIDs.Clear();

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
                pointersArray = pointers.ToArray();
            }

            for (int i = 0; i < pointers.Count; i++)
            {
                if (pointers[i] == -1)
                {
                    openIDs.Add(i);
                }
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

        public void Compact(int type = 0)
        {
            lock (this)
            {
                Stopwatch timer = Stopwatch.StartNew();
                Cleanup();

                if (Deallocations.Count == 0)
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
                    if (i != dislocations.Count - 1)
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
                for (int i = 0; i < Deallocations.Count; i++)
                {
                    SetPointer(Deallocations[i].id, -1);
                }

                CleanupPointers();

                Deallocations.Clear();
                currentCleanedUpLength = currentLength;
                
                timer.Stop();

                TraceCompaction(timer.Elapsed, type);

            }
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

    public class HVirtualAllocation<T> : IDisposable where T : unmanaged
    {
        private bool isDisposed;
        private VirtualMemory<T> backing;
        private VirtualAllocation<T> data;

        public HVirtualAllocation(VirtualMemory<T> backing, VirtualAllocation<T> data)
        {
            isDisposed = false;
            this.backing = backing;
            this.data = data;
        }

        public void Dispose()
        {
            if(!isDisposed)
            {
                isDisposed = true;
                backing.Deallocate(data);
            }
        }

        public VirtualAllocation<T> Get()
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

    public struct VirtualAllocation<T> where T : unmanaged
    {
        public long id;
        public long size;

        public VirtualAllocation(long size, long id)
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
