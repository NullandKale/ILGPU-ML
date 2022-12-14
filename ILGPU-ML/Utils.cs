using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ILGPU_ML
{
    public static class Utils
    {
        public static float Activate(float x, int method)
        {
            switch(method)
            {
                case 0:
                    return lrelu(x);
                case 1:
                    return sigmoid(x);
            }

            return sigmoid(x);
        }

        public static float dActivate(float x, int method)
        {
            switch (method)
            {
                case 0:
                    return dlrelu(x);
                case 1:
                    return dSigmoid(x);
            }

            return dSigmoid(x);
        }


        public static float lrelu(float x)
        {
            return (x > 0) ? x : 0.01f * x;
        }

        public static float dlrelu(float x)
        {
            return (x > 0) ? 1.0f : 0.01f;
        }

        public static float sigmoid(float x)
        {
            return 1f / (1f + MathF.Exp(-x));
        }

        public static float dSigmoid(float x)
        {
            return x * (1 - x);
        }

        public static int[] GenerateTrainingOrder(int size)
        {
            int[] order = new int[size];

            for(int i = 0; i < order.Length; i++)
            {
                order[i] = i;
            }

            return order;
        }

        public static void Shuffle<T>(Random rng, T[] array)
        {
            int n = array.Length;
            while (n > 1)
            {
                int k = rng.Next(n--);
                T temp = array[n];
                array[n] = array[k];
                array[k] = temp;
            }
        }

        public static int GetIndexOfMax(float[] data)
        {
            int currentMaxIndex = 0;
            float currentMax = data[0];

            for(int i = 0; i < data.Length; i++)
            {
                if(currentMax < data[i])
                {
                    currentMaxIndex = i;
                    currentMax = data[i];
                }
            }

            return currentMaxIndex;
        }

        internal static T[][] ToJaggedArray<T>(T[,] source)
        {
            int rowsFirstIndex = source.GetLowerBound(0);
            int rowsLastIndex = source.GetUpperBound(0);
            int numberOfRows = rowsLastIndex + 1;

            int columnsFirstIndex = source.GetLowerBound(1);
            int columnsLastIndex = source.GetUpperBound(1);
            int numberOfColumns = columnsLastIndex + 1;

            T[][] jaggedArray = new T[numberOfRows][];
            for (int i = rowsFirstIndex; i <= rowsLastIndex; i++)
            {
                jaggedArray[i] = new T[numberOfColumns];

                for (int j = columnsFirstIndex; j <= columnsLastIndex; j++)
                {
                    jaggedArray[i][j] = source[i, j];
                }
            }
            return jaggedArray;
        }

        public static T[,] To2D<T>(T[][] source)
        {
            try
            {
                int FirstDim = source.Length;
                int SecondDim = source.GroupBy(row => row.Length).Single().Key; // throws InvalidOperationException if source is not rectangular

                var result = new T[FirstDim, SecondDim];
                for (int i = 0; i < FirstDim; ++i)
                    for (int j = 0; j < SecondDim; ++j)
                        result[i, j] = source[i][j];

                return result;
            }
            catch (InvalidOperationException)
            {
                throw new InvalidOperationException("The given jagged array is not rectangular.");
            }
        }
    }
}
