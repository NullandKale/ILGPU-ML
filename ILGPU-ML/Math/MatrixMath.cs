using ILGPU;
using ILGPU.Runtime;
using ILGPU_ML.DataStructures;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace ILGPU_ML.Math
{
    public class MatrixMath
    {
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, float> InPlaceMul_M_F;
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, float> InPlaceDiv_M_F;
        private Action<Index2D, dVirtualMemory<float>, float, Matrix<float>> InPlaceDiv_F_M;
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, float> InPlaceAdd_M_F;
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, float> InPlaceSub_M_F;
        private Action<Index2D, dVirtualMemory<float>, float, Matrix<float>> InPlaceSub_F_M;

        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>> InPlaceElementWiseMul;
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>> InPlaceElementWiseDiv;
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>> InPlaceElementWiseAdd;
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>> InPlaceElementWiseSub;

        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>, Matrix<float>> ElementWiseMul;
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>, Matrix<float>> ElementWiseDiv;
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>, Matrix<float>> ElementWiseAdd;
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>, Matrix<float>> ElementWiseSub;

        private Action<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>, Matrix<float>> MatrixMul;
        
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>> MatrixSigmoid;
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>> MatrixRelu;
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>> MatrixDSigmoid;
        private Action<Index2D, dVirtualMemory<float>, Matrix<float>> MatrixDRelu;

        private Context context;
        private Accelerator device;

        private VirtualMemory<float> memory;

        public MatrixMath(Context context, Accelerator device, float maxMemoryPercent)
        {
            this.context = context;
            this.device = device;
            memory = new VirtualMemory<float>(device, maxMemoryPercent);

            InPlaceMul_M_F = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, float>(MatrixKernels.InPlaceMulKernel);
            InPlaceDiv_M_F = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, float>(MatrixKernels.InPlaceDivKernel);
            InPlaceDiv_F_M = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, float, Matrix<float>>(MatrixKernels.InPlaceDivKernel);
            InPlaceAdd_M_F = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, float>(MatrixKernels.InPlaceAddKernel);
            InPlaceSub_M_F = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, float>(MatrixKernels.InPlaceSubKernel);
            InPlaceSub_F_M = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, float, Matrix<float>>(MatrixKernels.InPlaceSubKernel);

            InPlaceElementWiseMul = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>>(MatrixKernels.ElementWiseMulKernel);
            InPlaceElementWiseDiv = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>>(MatrixKernels.ElementWiseDivKernel);
            InPlaceElementWiseAdd = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>>(MatrixKernels.ElementWiseAddKernel);
            InPlaceElementWiseSub = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>>(MatrixKernels.ElementWiseSubKernel);

            ElementWiseMul = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>, Matrix<float>>(MatrixKernels.ElementWiseMulKernel);
            ElementWiseDiv = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>, Matrix<float>>(MatrixKernels.ElementWiseDivKernel);
            ElementWiseAdd = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>, Matrix<float>>(MatrixKernels.ElementWiseAddKernel);
            ElementWiseSub = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>, Matrix<float>>(MatrixKernels.ElementWiseSubKernel);

            MatrixMul = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>, Matrix<float>, Matrix<float>>(MatrixKernels.MatrixMulKernel);

            MatrixSigmoid = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>>(MatrixKernels.MatrixSigmoidKernel);
            MatrixRelu = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>>(MatrixKernels.MatrixReluKernel);
            MatrixDSigmoid = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>>(MatrixKernels.MatrixDSigmoidKernel);
            MatrixDRelu = device.LoadAutoGroupedStreamKernel<Index2D, dVirtualMemory<float>, Matrix<float>>(MatrixKernels.MatrixDReluKernel);
        }

        public HMatrix AllocateMatrix(Vec2i size)
        {
            return new HMatrix(memory, size);
        }

        public HMatrix AllocateMatrix(Vec2i size, float[,] data)
        {
            if(data.GetLength(0) != size.x || data.GetLength(1) != size.y)
            {
                throw new ArgumentOutOfRangeException("array size must match size");
            }

            HMatrix mat = new HMatrix(memory, size);

            mat.GetArrayView().CopyFromCPU(data);

            return mat;
        }

        public void MulInPlace(HMatrix matrix, float scalar) 
        {
            InPlaceMul_M_F(matrix.Get().size, memory.GetD(), matrix.Get(), scalar);
            device.Synchronize();
        }

        public void MulInPlace(HMatrix matrixA, HMatrix matrixB)
        {
            InPlaceElementWiseMul(matrixA.Get().size, memory.GetD(), matrixA.Get(), matrixB.Get());
            device.Synchronize();
        }

        public void DivInPlace(HMatrix matrix, float scalar)
        {
            InPlaceDiv_M_F(matrix.Get().size, memory.GetD(), matrix.Get(), scalar);
            device.Synchronize();
        }

        public void DivInPlace(float scalar, HMatrix matrix)
        {
            InPlaceDiv_F_M(matrix.Get().size, memory.GetD(), scalar, matrix.Get());
            device.Synchronize();
        }

        public void DivInPlace(HMatrix matrixA, HMatrix matrixB)
        {
            InPlaceElementWiseDiv(matrixA.Get().size, memory.GetD(), matrixA.Get(), matrixB.Get());
            device.Synchronize();
        }

        public void AddInPlace(HMatrix matrix, float scalar)
        {
            InPlaceAdd_M_F(matrix.Get().size, memory.GetD(), matrix.Get(), scalar);
            device.Synchronize();
        }

        public void AddInPlace(HMatrix matrixA, HMatrix matrixB)
        {
            InPlaceElementWiseAdd(matrixA.Get().size, memory.GetD(), matrixA.Get(), matrixB.Get());
            device.Synchronize();
        }

        public void SubInPlace(HMatrix matrix, float scalar)
        {
            InPlaceSub_M_F(matrix.Get().size, memory.GetD(), matrix.Get(), scalar);
            device.Synchronize();
        }

        public void SubInPlace(float scalar, HMatrix matrix)
        {
            InPlaceSub_F_M(matrix.Get().size, memory.GetD(), scalar, matrix.Get());
            device.Synchronize();
        }

        public void SubInPlace(HMatrix matrixA, HMatrix matrixB)
        {
            InPlaceElementWiseSub(matrixA.Get().size, memory.GetD(), matrixA.Get(), matrixB.Get());
            device.Synchronize();
        }

        public HMatrix Mul(HMatrix matrixA, HMatrix matrixB)
        {
            HMatrix matrixC = AllocateMatrix(matrixA.Get().size);

            ElementWiseMul(matrixA.Get().size, memory.GetD(), matrixA.Get(), matrixB.Get(), matrixC.Get());
            device.Synchronize();

            return matrixC;
        }

        public HMatrix Div(HMatrix matrixA, HMatrix matrixB)
        {
            HMatrix matrixC = AllocateMatrix(matrixA.Get().size);

            ElementWiseDiv(matrixA.Get().size, memory.GetD(), matrixA.Get(), matrixB.Get(), matrixC.Get());
            device.Synchronize();

            return matrixC;
        }

        public HMatrix Add(HMatrix matrixA, HMatrix matrixB)
        {
            HMatrix matrixC = AllocateMatrix(matrixA.Get().size);

            ElementWiseAdd(matrixA.Get().size, memory.GetD(), matrixA.Get(), matrixB.Get(), matrixC.Get());
            device.Synchronize();

            return matrixC;
        }

        public HMatrix Sub(HMatrix matrixA, HMatrix matrixB)
        {
            HMatrix matrixC = AllocateMatrix(matrixA.Get().size);

            ElementWiseSub(matrixA.Get().size, memory.GetD(), matrixA.Get(), matrixB.Get(), matrixC.Get());
            device.Synchronize();

            return matrixC;
        }

        public HMatrix MatMul(HMatrix matrixA, HMatrix matrixB)
        {
            HMatrix matrixC = AllocateMatrix(matrixA.Get().size);

            MatrixMul(matrixA.Get().size, memory.GetD(), matrixA.Get(), matrixB.Get(), matrixC.Get());
            device.Synchronize();

            return matrixC;
        }

        public void Sigmoid(HMatrix matrix)
        {
            MatrixSigmoid(matrix.Get().size, memory.GetD(), matrix.Get());
            device.Synchronize();
        }

        public void dSigmoid(HMatrix matrix)
        {
            MatrixDSigmoid(matrix.Get().size, memory.GetD(), matrix.Get());
            device.Synchronize();
        }

        public void Relu(HMatrix matrix)
        {
            MatrixRelu(matrix.Get().size, memory.GetD(), matrix.Get());
            device.Synchronize();
        }

        public void dRelu(HMatrix matrix)
        {
            MatrixDRelu(matrix.Get().size, memory.GetD(), matrix.Get());
            device.Synchronize();
        }
    }
}
