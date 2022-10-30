#include "./EasyBMP/EasyBMP.h"
#include <iostream>
#include <chrono>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace chrono;

//-------------------Parametrs-------------------

int BitDepth = 8;  // Глубина цвета (1, 4, 8)
string inputPath = "original.bmp";  // Путь к изначальному изображению (C://original.bmp либо, если нужно в туже папку, где и код original.bmp)
string outputPath = "after.bmp";  // Путь к обработанному изображению (C://after.bmp либо, если нужно в туже папку, где и код after.bmp)

const char* charInput = inputPath.c_str();  // Переменные для перевода в constChar
const char* charOutput = outputPath.c_str();

texture<int, 2, cudaReadModeElementType> texRef;  // 2D текстура для Cuda

system_clock::time_point start_time, end_time_input, end_time_load_memory,
                          end_time_filter, end_time_output, end_time;

//-------------------Functions-------------------

__global__ void meanfilter_kernel(int* output, int width)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int Temp[9] = { tex2D(texRef, x - 1, y - 1), tex2D(texRef, x, y - 1), tex2D(texRef, x + 1, y - 1),
        tex2D(texRef, x - 1, y), tex2D(texRef, x, y), tex2D(texRef, x + 1, y),
        tex2D(texRef, x - 1, y + 1), tex2D(texRef, x, y + 1), tex2D(texRef, x + 1, y + 1) };

    for (int i = 0; i < 9; i++)
    {
        int temp = 0; // Переменная для хранения временного значения
        for (int j = i; j < 9; j++)
        {
            if (Temp[j] < Temp[i])
            {
                temp = Temp[i];
                Temp[i] = Temp[j];
                Temp[j] = temp;
            }
        }
    }

    output[y * width + x] = Temp[4];
}

void  CreateGreyColorTable(BMP & InputImage)  // Создание серого градиента (поддерживается до 8 бит)
{
    int BitDepth = InputImage.TellBitDepth(); // Узнаем глубину цвета изображения
    int NumberOfColors = InputImage.TellNumberOfColors();  // Узнаем колличество цветов
    ebmpBYTE StepSize; // Размер шага
    if (BitDepth != 1) // Если глубина цвета не равна одному
    {
        StepSize = 255 / (NumberOfColors - 1);  // Вычисляем шаг
    }
    else  //Иначе шаг равен 255
    {
        StepSize = 255;
    }
    for (int i = 0; i < NumberOfColors; i++) // Заполняем каждый бит серым цветом  шагом
    {
        RGBApixel Temp;
        Temp.Red = i * StepSize;
        Temp.Green = i * StepSize;
        Temp.Blue = i * StepSize;
        Temp.Alpha = 0;
        InputImage.SetColor(i, Temp); // И добавляем его в палитру
    }
}

// Функция вывода времени и текста
string outputTime(string title, system_clock::time_point start, system_clock::time_point end)
{
    if (duration_cast<nanoseconds>(end - start).count() > 3600000000000)
    {
        title += to_string(duration_cast<nanoseconds>(end - start).count() / 3600000000000);
        title += ".";
        title += to_string(duration_cast<nanoseconds>(end - start).count() -
            (duration_cast<nanoseconds>(end - start).count() / 3600000000000) * 3600000000000);
        title += " hours";
    }
    else if (duration_cast<nanoseconds>(end - start).count() > 60000000000)
    {
        title += to_string(duration_cast<nanoseconds>(end - start).count() / 60000000000);
        title += ".";
        title += to_string(duration_cast<nanoseconds>(end - start).count() -
            (duration_cast<nanoseconds>(end - start).count() / 60000000000) * 60000000000);
        title += " min";
    }
    else if (duration_cast<nanoseconds>(end - start).count() > 1000000000)
    {
        title += to_string(duration_cast<nanoseconds>(end - start).count() / 1000000000);
        title += ".";
        title += to_string(duration_cast<nanoseconds>(end - start).count() -
            (duration_cast<nanoseconds>(end - start).count() / 1000000000) * 1000000000);
        title += " s";
    }
    else if (duration_cast<nanoseconds>(end - start).count() > 1000000)
    {
        title += to_string(duration_cast<nanoseconds>(end - start).count() / 1000000);
        title += ".";
        title += to_string(duration_cast<nanoseconds>(end - start).count() -
            (duration_cast<nanoseconds>(end - start).count() / 1000000) * 1000000);
        title += " ms";
    }
    else if (duration_cast<nanoseconds>(end - start).count() > 1000)
    {
        title += to_string(duration_cast<nanoseconds>(end - start).count() / 1000);
        title += ".";
        title += to_string(duration_cast<nanoseconds>(end - start).count() -
            (duration_cast<nanoseconds>(end - start).count() / 1000) * 1000);
        title += " us";
    }
    else
    {
        title += to_string(duration_cast<nanoseconds>(end - start).count());
        title += " ns";
    }

    return title;
}

string outlineTime(string title, system_clock::time_point start, system_clock::time_point end)
{
    string temp = "";

    title = outputTime(title, start, end);

    for (int i = 0; i < title.length(); i++)
        temp += "-";

    return temp;
}

//-----------------------------------------------


int main()
{
    start_time = system_clock::now(); // Начальное времени программы

    BMP beforeImage;  // Создаем изображение
    beforeImage.ReadFromFile(charInput);  // Подгружаем его
    beforeImage.SetBitDepth(BitDepth);  // Задаем глубину цвета

    CreateGreyColorTable(beforeImage);  // Устанавливаем цветовую гамму в градациях серого в зависимости от глубины цвета изображения

    end_time_input = system_clock::now(); // Конец времени загрузки

    // Создаем массив для цветов изображения
    size_t imageSize = beforeImage.TellWidth() * beforeImage.TellHeight() * sizeof(int);  // узнаем необходимый размер для массива
    int* colorGrey = (int*)malloc(imageSize);  // Выделяем память под массив

    // Заполняем массив номером цвета
    for (int i = 0; i < beforeImage.TellHeight(); i++) // По y
        for (int j = 0; j < beforeImage.TellWidth(); j++) // По x
            colorGrey[i * beforeImage.TellWidth() + j] = (int)beforeImage(j, i)->Red;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
    cudaArray* srcArray;
    cudaMallocArray(&srcArray, &channelDesc, beforeImage.TellWidth(), beforeImage.TellHeight());

    // Помещаем копию изоюражения в память
    cudaMemcpyToArray(srcArray, 0, 0, colorGrey, beforeImage.TellWidth() * beforeImage.TellHeight() * sizeof(int), cudaMemcpyHostToDevice);

    // Привязываем память к текстуре
    cudaBindTextureToArray(&texRef, srcArray, &channelDesc);

    // Выделяем память в device
    int* output;
    cudaMalloc(&output, beforeImage.TellWidth() * beforeImage.TellHeight() * sizeof(int));

    end_time_load_memory = system_clock::now(); // Конец времени выделения памяти

    // Запуск kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((beforeImage.TellWidth() + dimBlock.x - 1) / dimBlock.x,
        (beforeImage.TellHeight() + dimBlock.y - 1) / dimBlock.y);
    meanfilter_kernel << < dimGrid, dimBlock >> > (output, beforeImage.TellWidth());

    // Синхронизация потоков
    cudaThreadSynchronize();

    end_time_filter = system_clock::now(); // Конец времени загрузки

    // Передаем данные обратно к host
    cudaMemcpy(colorGrey, output, beforeImage.TellWidth() * beforeImage.TellHeight() * sizeof(int), cudaMemcpyDeviceToHost);

    // Загружаем данные на изображение
    for (int i = 0; i < beforeImage.TellHeight(); i++)
    {
        for (int j = 0; j < beforeImage.TellWidth(); j++)
        {
            beforeImage(j, i)->Red = colorGrey[i * beforeImage.TellWidth() + j];
            beforeImage(j, i)->Green = colorGrey[i * beforeImage.TellWidth() + j];
            beforeImage(j, i)->Blue = colorGrey[i * beforeImage.TellWidth() + j];
        }
    }

    beforeImage.WriteToFile(charOutput);  // Выгружаем данные

    end_time_output = system_clock::now(); // Конец выгрузки изображения

    //Отчистка памяти
    cudaUnbindTexture(&texRef);
    cudaFreeArray(srcArray);
    cudaFree(output);
    free(colorGrey);

    end_time = system_clock::now(); // Конечное время программы

    cout << outputTime("Total program running time: ", start_time, end_time) << endl;
    cout << outlineTime("Total program running time: ", start_time, end_time) << endl;
    cout << outputTime("Image loading time: ", start_time, end_time_input) << endl;
    cout << outputTime("Memory allocation time: ", end_time_input, end_time_load_memory) << endl;
    cout << outputTime("Filter application time: ", end_time_load_memory, end_time_filter) << endl;
    cout << outputTime("Image uploading time: ", end_time_filter, end_time_output) << endl;
    cout << outputTime("Memory release time: ", end_time_output, end_time) << endl;
    cout << outlineTime("Total program running time: ", start_time, end_time) << endl;

    return 0;
}