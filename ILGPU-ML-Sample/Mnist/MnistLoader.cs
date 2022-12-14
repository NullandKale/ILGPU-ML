using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;

namespace ILGPU_ML
{
    public static class MnistReader
    {
        public const int width = 28;
        public const int height = 28;

        private const string TrainImages = "mnist/train-images.idx3-ubyte";
        private const string TrainLabels = "mnist/train-labels.idx1-ubyte";
        private const string TestImages = "mnist/t10k-images.idx3-ubyte";
        private const string TestLabels = "mnist/t10k-labels.idx1-ubyte";

        public static List<(float[] data, float[] label)> ReadTrainingData()
        {
            return Read(TrainImages, TrainLabels);
        }

        public static List<(float[] data, float[] label)> ReadTestData()
        {
            return Read(TestImages, TestLabels);
        }

        private static List<(float[] data, float[] label)> Read(string imagesPath, string labelsPath)
        {
            List<(float[] data, float[] label)> data = new List<(float[] data, float[] label)>();

            using BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            using BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < numberOfImages; i++)
            {
                byte[] bytes = images.ReadBytes(width * height);
                byte label = labels.ReadByte();

                float[] floatData = new float[bytes.Length];
                float[] labelData = new float[10];

                for(int j = 0; j < bytes.Length; j++)
                {
                    floatData[j] = (((float)bytes[j]) - (127.5f)) / 127.5f;
                }

                for(int j = 0; j < 10; j++)
                {
                    if(j == label)
                    {
                        labelData[j] = 1;
                    }
                    else
                    {
                        labelData[j] = 0.01f;
                    }
                }

                data.Add((floatData, labelData));
            }

            return data;
        }
    }

    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
