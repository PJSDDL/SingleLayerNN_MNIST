#include <stdio.h>
#include <stdlib.h>

#include "WEIGHTS.h"


void printMNIST(float MNIST[], int height)
{
    for (int x = 0; x < height; x++)
    {
        for (int y = 0; y < height; y++)
        {
            if (MNIST[x * height + y] < 1E-1)
            {
                printf("_");
            }
            else
            {
                printf("@");
            }
        }
        printf("\n");
    }
}

//单层神经网络，输入16*16矩阵，输出10种数字分类
int SingleLayerNN(float MNIST[], float W[], int height, float b[], int num)
{
    float labels[10];

    for (int index = 0; index < num; index++)
    {
        labels[index] = 0;
    }

    for (int index = 0; index < num; index++)
    {
        for (int i = 0; i < height*height; i++)
        {
            labels[index] += W[num * i + index] * MNIST[i];
        }

        printf("%f ", labels[index] - b[index]);
    }
    printf("\n");

    int index_max = 0;
    for (int index = 0; index < num; index++)
    {
        if (labels[index] > labels[index_max])
        {
            index_max = index;
        }
    }

    printf("index_max: %d \n", index_max);
    return index_max;
}

int main()
{
    printMNIST(NUM1, 16);
    SingleLayerNN(NUM1, W, 16, b, 10);
    printMNIST(NUM2, 16);
    SingleLayerNN(NUM2, W, 16, b, 10);
    printMNIST(NUM3, 16);
    SingleLayerNN(NUM3, W, 16, b, 10);
    printMNIST(NUM4, 16);
    SingleLayerNN(NUM4, W, 16, b, 10);
    printMNIST(NUM5, 16);
    SingleLayerNN(NUM5, W, 16, b, 10);
    printMNIST(NUM6, 16);
    SingleLayerNN(NUM6, W, 16, b, 10);

    return 0;
}
