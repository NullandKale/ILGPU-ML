using ILGPU;
using ILGPU_ML.DataStructures;

namespace ILGPU_ML.Math
{
    public static class MatrixKernels
    {
        public static void InPlaceMulKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrix, float scalar)
        {
            float val = matrix.Get(data, index);
            matrix.Set(data, index, val * scalar);
        }

        public static void InPlaceDivKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrix, float scalar)
        {
            float val = matrix.Get(data, index);
            matrix.Set(data, index, val / scalar);
        }

        public static void InPlaceDivKernel(Index2D index, dVirtualMemory<float> data, float scalar, Matrix<float> matrix)
        {
            float val = matrix.Get(data, index);
            matrix.Set(data, index, scalar / val);
        }

        public static void InPlaceAddKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrix, float scalar)
        {
            float val = matrix.Get(data, index);
            matrix.Set(data, index, val + scalar);
        }

        public static void InPlaceSubKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrix, float scalar)
        {
            float val = matrix.Get(data, index);
            matrix.Set(data, index, val - scalar);
        }

        public static void InPlaceSubKernel(Index2D index, dVirtualMemory<float> data, float scalar, Matrix<float> matrix)
        {
            float val = matrix.Get(data, index);
            matrix.Set(data, index, scalar - val);
        }

        public static void ElementWiseMulKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA, Matrix<float> matrixB)
        {
            float valA = matrixA.Get(data, index);
            float valB = matrixB.Get(data, index);
            matrixA.Set(data, index, valA * valB);
        }

        public static void ElementWiseDivKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA, Matrix<float> matrixB)
        {
            float valA = matrixA.Get(data, index);
            float valB = matrixB.Get(data, index);
            matrixA.Set(data, index, valA / valB);
        }

        public static void ElementWiseAddKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA, Matrix<float> matrixB)
        {
            float valA = matrixA.Get(data, index);
            float valB = matrixB.Get(data, index);
            matrixA.Set(data, index, valA + valB);
        }

        public static void ElementWiseSubKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA, Matrix<float> matrixB)
        {
            float valA = matrixA.Get(data, index);
            float valB = matrixB.Get(data, index);
            matrixA.Set(data, index, valA - valB);
        }

        public static void ElementWiseMulKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA, Matrix<float> matrixB, Matrix<float> matrixC)
        {
            float valA = matrixA.Get(data, index);
            float valB = matrixB.Get(data, index);
            matrixC.Set(data, index, valA * valB);
        }

        public static void ElementWiseDivKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA, Matrix<float> matrixB, Matrix<float> matrixC)
        {
            float valA = matrixA.Get(data, index);
            float valB = matrixB.Get(data, index);
            matrixC.Set(data, index, valA / valB);
        }

        public static void ElementWiseAddKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA, Matrix<float> matrixB, Matrix<float> matrixC)
        {
            float valA = matrixA.Get(data, index);
            float valB = matrixB.Get(data, index);
            matrixC.Set(data, index, valA + valB);
        }

        public static void ElementWiseSubKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA, Matrix<float> matrixB, Matrix<float> matrixC)
        {
            float valA = matrixA.Get(data, index);
            float valB = matrixB.Get(data, index);
            matrixC.Set(data, index, valA - valB);
        }

        public static void MatrixMulKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA, Matrix<float> matrixB, Matrix<float> matrixC)
        {
            float sum = 0;
            for (int k = 0; k < matrixA.data.size.y; k++)
            {
                float valA = matrixA.Get(data, index.X, k);
                float valB = matrixB.Get(data, k, index.Y);
                sum += valA * valB;
            }

            matrixC.Set(data, index, sum);
        }

        public static void MatrixSigmoidKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA)
        {
            float valA = matrixA.Get(data, index);
            matrixA.Set(data, index, Utils.sigmoid(valA));
        }

        public static void MatrixReluKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA)
        {
            float valA = matrixA.Get(data, index);
            matrixA.Set(data, index, Utils.lrelu(valA));
        }

        public static void MatrixDSigmoidKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA)
        {
            float valA = matrixA.Get(data, index);
            matrixA.Set(data, index, Utils.dSigmoid(valA));
        }

        public static void MatrixDReluKernel(Index2D index, dVirtualMemory<float> data, Matrix<float> matrixA)
        {
            float valA = matrixA.Get(data, index);
            matrixA.Set(data, index, Utils.dlrelu(valA));
        }
    }
}
