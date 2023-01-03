﻿using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU_ML.Math;
using ILGPU_ML.DataStructures;

namespace ILGPU_ML.MatrixTests
{
    public static class MatrixTestSample
    {
        private static Context context;
        private static Accelerator device;

        public static void PrintArray(float[,] a, Vec2i size)
        {
            for (int i = 0; i < size.x; i++)
            {
                for (int j = 0; j < size.y; j++)
                {
                    Console.Write(a[i, j] + " ");
                }
                Console.WriteLine();
            }

            Console.WriteLine();
        }

        public static bool ArraysEqual(float[,] a, float[,] b)
        {
            if(a.GetLength(0) != b.GetLength(0))
            {
                return false;
            }

            if (a.GetLength(1) != b.GetLength(1))
            {
                return false;
            }

            for(int i = 0; i < a.GetLength(0); i++)
            {
                for(int j = 0; j < b.GetLength(1); j++)
                {
                    if (a[i, j] != b[i, j])
                    {
                        return false;
                    }

                }
            }

            return true;
        }

        public static void CheckAddInPlace_M_M(MatrixMath math)
        {
            float[,] input = new float[,] { { 1, 1 }, { 1, 1 } };
            float[,] output = new float[,] { { 2, 2 }, { 2, 2 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.AddInPlace(a, a);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix += Matrix Checked!");
            }
            else
            {
                throw new Exception("Matrix += Matrix Failed!");
            }
        }

        public static void CheckAddInPlace_M_S(MatrixMath math)
        {
            float[,] input = new float[,] { { 0, 0 }, { 0, 0 } };
            float[,] output = new float[,] { { 2, 2 }, { 2, 2 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.AddInPlace(a, 2);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix += Scalar Checked!");
            }
            else
            {
                throw new Exception("Matrix += Scalar Failed!");
            }
        }

        public static void CheckSubInPlace_M_M(MatrixMath math)
        {
            float[,] input = new float[,] { { 1, 1 }, { 1, 1 } };
            float[,] output = new float[,] { { 0, 0 }, { 0, 0 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.SubInPlace(a, a);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix -= Matrix Checked!");
            }
            else
            {
                throw new Exception("Matrix -= Matrix Failed!");
            }
        }

        public static void CheckSubInPlace_M_S(MatrixMath math)
        {
            float[,] input = new float[,] { { 4, 4 }, { 4, 4 } };
            float[,] output = new float[,] { { 0, 0 }, { 0, 0 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.SubInPlace(a, 4);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix -= Scalar Checked!");
            }
            else
            {
                throw new Exception("Matrix -= Scalar Failed!");
            }
        }

        public static void CheckSubInPlace_S_M(MatrixMath math)
        {
            float[,] input = new float[,] { { 2, 2 }, { 2, 2 } };
            float[,] output = new float[,] { {0, 0 }, { 0, 0 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.SubInPlace(2, a);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Scalar -= Matrix Checked!");
            }
            else
            {
                throw new Exception("Scalar -= Matrix Failed!");
            }
        }

        public static void CheckMulInPlace_M_M(MatrixMath math)
        {
            float[,] input = new float[,] { { 2, 2 }, { 2, 2 } };
            float[,] output = new float[,] { { 4, 4 }, { 4, 4 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.MulInPlace(a, a);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix *= Matrix Checked!");
            }
            else
            {
                throw new Exception("Matrix *= Matrix Failed!");
            }
        }

        public static void CheckMulInPlace_M_S(MatrixMath math)
        {
            float[,] input = new float[,] { { 1, 1 }, { 1, 1 } };
            float[,] output = new float[,] { { 2, 2 }, { 2, 2 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.MulInPlace(a, 2);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix *= Scalar Checked!");
            }
            else
            {
                throw new Exception("Matrix *= Scalar Failed!");
            }
        }

        public static void CheckDivInPlace_M_M(MatrixMath math)
        {
            float[,] input = new float[,] { { 4, 4 }, { 2, 2 } };
            float[,] output = new float[,] { {1, 1 }, { 1, 1 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.DivInPlace(a, a);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix /= Matrix Checked!");
            }
            else
            {
                throw new Exception("Matrix /= Matrix Failed!");
            }
        }

        public static void CheckDivInPlace_M_S(MatrixMath math)
        {
            float[,] input = new float[,] { { 4, 4 }, { 4, 4 } };
            float[,] output = new float[,] { { 2, 2 }, { 2, 2 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.DivInPlace(a, 2);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix /= Scalar Checked!");
            }
            else
            {
                throw new Exception("Matrix /= Scalar Failed!");
            }
        }

        public static void CheckDivInPlace_S_M(MatrixMath math)
        {
            float[,] input = new float[,] { { 2, 2 }, { 2, 2 } };
            float[,] output = new float[,] { { 5, 5 }, { 5, 5 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.DivInPlace(10, a);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Scalar /= Matrix Checked!");
            }
            else
            {
                throw new Exception("Scalar /= Matrix Failed!");
            }
        }

        public static void CheckAdd(MatrixMath math)
        {
            float[,] inputA = new float[,] { { 2, 2 }, { 2, 2 } };
            float[,] inputB = new float[,] { { 3, 3 }, { 3, 3 } };
            float[,] output = new float[,] { { 5, 5 }, { 5, 5 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), inputA);
            using HMatrix b = math.AllocateMatrix(new Vec2i(2, 2), inputB);

            using HMatrix c = math.Add(a, b);

            float[,] data = c.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix = Matrix + Matrix Checked!");
            }
            else
            {
                throw new Exception("Matrix = Matrix + Matrix Failed!");
            }
        }

        public static void CheckSub(MatrixMath math)
        {
            float[,] inputA = new float[,] { { 2, 2 }, { 2, 2 } };
            float[,] inputB = new float[,] { { 3, 3 }, { 3, 3 } };
            float[,] output = new float[,] { { -1, -1 }, { -1, -1 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), inputA);
            using HMatrix b = math.AllocateMatrix(new Vec2i(2, 2), inputB);

            using HMatrix c = math.Sub(a, b);

            float[,] data = c.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix = Matrix - Matrix Checked!");
            }
            else
            {
                throw new Exception("Matrix = Matrix - Matrix Failed!");
            }
        }

        public static void CheckMul(MatrixMath math)
        {
            float[,] inputA = new float[,] { { 2, 2 }, { 2, 2 } };
            float[,] inputB = new float[,] { { 3, 3 }, { 3, 3 } };
            float[,] output = new float[,] { { 6, 6 }, { 6, 6 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), inputA);
            using HMatrix b = math.AllocateMatrix(new Vec2i(2, 2), inputB);

            using HMatrix c = math.Mul(a, b);

            float[,] data = c.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix = Matrix * Matrix Checked!");
            }
            else
            {
                throw new Exception("Matrix = Matrix * Matrix Failed!");
            }
        }

        public static void CheckDiv(MatrixMath math)
        {
            float[,] inputA = new float[,] { { 6, 6 }, { 6, 6 } };
            float[,] inputB = new float[,] { { 3, 3 }, { 3, 3 } };
            float[,] output = new float[,] { { 2, 2 }, { 2, 2 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), inputA);
            using HMatrix b = math.AllocateMatrix(new Vec2i(2, 2), inputB);

            using HMatrix c = math.Div(a, b);

            float[,] data = c.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix = Matrix / Matrix Checked!");
            }
            else
            {
                throw new Exception("Matrix = Matrix / Matrix Failed!");
            }
        }

        public static void CheckSigmoid(MatrixMath math)
        {
            float[,] input = new float[,] { { 2, 2 }, { 2, 2 } };
            float[,] output = new float[,] { { 0.880797f, 0.880797f }, { 0.880797f, 0.880797f } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.Sigmoid(a);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix = Sigmoid(Matrix) Checked!");
            }
            else
            {
                PrintArray(data, a.Get().size);
                throw new Exception("Matrix = Sigmoid(Matrix) Failed!");
            }
        }

        public static void CheckdSigmoid(MatrixMath math)
        {
            float[,] input = new float[,] { { 2, 2 }, { 2, 2 } };
            float[,] output = new float[,] { { -2, -2 }, { -2, -2 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.dSigmoid(a);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix = dSigmoid(Matrix) Checked!");
            }
            else
            {
                PrintArray(data, a.Get().size);
                throw new Exception("Matrix = dSigmoid(Matrix) Failed!");
            }
        }

        public static void CheckRelu(MatrixMath math)
        {
            float[,] input = new float[,] { { 2, 2 }, { 2, 2 } };
            float[,] output = new float[,] { { 2, 2 }, { 2, 2 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.Relu(a);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix = Relu(Matrix) Checked!");
            }
            else
            {
                PrintArray(data, a.Get().size);
                throw new Exception("Matrix = Relu(Matrix) Failed!");
            }
        }

        public static void CheckdRelu(MatrixMath math)
        {
            float[,] input = new float[,] { { 2, 2 }, { 2, 2 } };
            float[,] output = new float[,] { { 1, 1 }, { 1, 1 } };

            using HMatrix a = math.AllocateMatrix(new Vec2i(2, 2), input);

            math.dRelu(a);

            float[,] data = a.GetCPUData();

            if (ArraysEqual(data, output))
            {
                Console.WriteLine("Matrix = dRelu(Matrix) Checked!");
            }
            else
            {
                PrintArray(data, a.Get().size);
                throw new Exception("Matrix = dRelu(Matrix) Failed!");
            }
        }


        public static void Run()
        {
            bool debug = false;
            context = Context.Create(builder => builder.CPU().Cuda().EnableAlgorithms().Optimize(debug ? OptimizationLevel.Debug : OptimizationLevel.O1));
            device = context.GetPreferredDevice(preferCPU: false)
                                      .CreateAccelerator(context);

            MatrixMath math = new MatrixMath(context, device, 0.95f);

            CheckAddInPlace_M_M(math);
            CheckAddInPlace_M_S(math);
            CheckAdd(math);

            CheckSubInPlace_M_M(math);
            CheckSubInPlace_M_S(math);
            CheckSubInPlace_S_M(math);
            CheckSub(math);

            CheckMulInPlace_M_M(math);
            CheckMulInPlace_M_S(math);
            CheckMul(math);

            CheckDivInPlace_M_M(math);
            CheckDivInPlace_M_S(math);
            CheckDivInPlace_S_M(math);
            CheckDiv(math);

            CheckSigmoid(math);
            CheckdSigmoid(math);

            CheckRelu(math);
            CheckdRelu(math);

            Console.ReadLine();
        }
    }
}
