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

        private VirtualMemory<float> data;

        public MatrixMath(Context context, Accelerator device, float maxMemoryPercent)
        {
            this.context = context;
            this.device = device;
            data = new VirtualMemory<float>(device, maxMemoryPercent);

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

        public Matrix<float> AllocateMatrix(Vec2i size)
        {
            return new Matrix<float>(data, size);
        }

        public Vector<float> AllocateVector(int size)
        {
            return new Vector<float>(data, size);
        }

        public void MulInPlace(Matrix<float> matrix, float scalar) 
        {
            InPlaceMul_M_F(matrix.data.size, data.GetD(), matrix, scalar);
        }

        public void MulInPlace(Matrix<float> matrixA, Matrix<float> matrixB)
        {
            InPlaceElementWiseMul(matrixA.data.size, data.GetD(), matrixA, matrixB);
        }

        public void DivInPlace(Matrix<float> matrix, float scalar)
        {
            InPlaceDiv_M_F(matrix.data.size, data.GetD(), matrix, scalar);
        }

        public void DivInPlace(float scalar, Matrix<float> matrix)
        {
            InPlaceDiv_F_M(matrix.data.size, data.GetD(), scalar, matrix);
        }

        public void DivInPlace(Matrix<float> matrixA, Matrix<float> matrixB)
        {
            InPlaceElementWiseDiv(matrixA.data.size, data.GetD(), matrixA, matrixB);
        }

        public void AddInPlace(Matrix<float> matrix, float scalar)
        {
            InPlaceAdd_M_F(matrix.data.size, data.GetD(), matrix, scalar);
        }

        public void AddInPlace(Matrix<float> matrixA, Matrix<float> matrixB)
        {
            InPlaceElementWiseAdd(matrixA.data.size, data.GetD(), matrixA, matrixB);
        }

        public void SubInPlace(Matrix<float> matrix, float scalar)
        {
            InPlaceSub_M_F(matrix.data.size, data.GetD(), matrix, scalar);
        }

        public void SubInPlace(float scalar, Matrix<float> matrix)
        {
            InPlaceSub_F_M(matrix.data.size, data.GetD(), scalar, matrix);
        }

        public void SubInPlace(Matrix<float> matrixA, Matrix<float> matrixB)
        {
            InPlaceElementWiseSub(matrixA.data.size, data.GetD(), matrixA, matrixB);
        }
    }
}
