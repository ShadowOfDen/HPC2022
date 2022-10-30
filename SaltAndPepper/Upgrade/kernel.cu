#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <string>

using namespace cv;
using namespace std;
using namespace chrono;

//-------------------Parametrs-------------------

string inputPath = "data//input1_noise_02.jpg";  // Путь к изначальному изображению (C://original.bmp либо, если нужно в туже папку, где и код original.bmp)
string outputPath = "outputData//output1_noise_02.jpg";  // Путь к обработанному изображению (C://after.bmp либо, если нужно в туже папку, где и код after.bmp)

bool timeText = true;  // Переменная для отображения текста времени выполнения работы (true/false - вкл/выкл)

int filterSize = 3;  // Размер фильтра (filterSize * filterSize) Максимум 5 на 5!!!

//----------------Global-Parametrs---------------

texture <uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> cudaTexture;  // Создаем 2D текстуру

// Метки времени
system_clock::time_point start_time, end_time_input, end_time_load_memory,
                          end_time_filter, end_time_output, end_time;

//-------------------Functions-------------------

__global__ void meanFilter(uchar* deviceData, int imgHeight, int imgWidth, int channels, int filterSize)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;  // Координата x
    const int y = blockDim.y * blockIdx.y + threadIdx.y;  // Координата y
    int yWx = (y * imgWidth + x) * channels;  // Координата одномерного массива

    float TempB[25];  // Инициализируем массивы для фильтрации (максимум 5 на 5)
    float TempG[25];
    float TempR[25];

    for (int i = 0; i < 25; i++)
    {
        TempB[i] = 0;
        TempG[i] = 0;
        TempR[i] = 0;
    }

    // Заполняем массивы
    for (int i = 0; i < filterSize; i++)
    {
        for (int j = 0; j < filterSize; j++)
        {
            if (filterSize % 2 != 0)  // Если фильтр нечетный
            {
                TempB[i * filterSize + j] = tex2D(cudaTexture, x + j - (filterSize - 1) / 2, y + i - (filterSize - 1) / 2).x;
                TempG[i * filterSize + j] = tex2D(cudaTexture, x + j - (filterSize - 1) / 2, y + i - (filterSize - 1) / 2).y;
                TempR[i * filterSize + j] = tex2D(cudaTexture, x + j - (filterSize - 1) / 2, y + i - (filterSize - 1) / 2).z;
            }
            else  // Если фильтр четный
            {
                TempB[i * filterSize + j] = tex2D(cudaTexture, x + j - filterSize / 2, y + i - filterSize / 2).x;
                TempG[i * filterSize + j] = tex2D(cudaTexture, x + j - filterSize / 2, y + i - filterSize / 2).y;
                TempR[i * filterSize + j] = tex2D(cudaTexture, x + j - filterSize / 2, y + i - filterSize / 2).z;
            }
        }
    }

    // Фильтрация массивов цветов
    for (int i = 0; i < filterSize * filterSize; i++)
    {
        float temp = 0; // Переменная для хранения временного значения
        for (int j = i; j < filterSize * filterSize; j++)
        {
            if (TempB[j] < TempB[i])
            {
                temp = TempB[i];
                TempB[i] = TempB[j];
                TempB[j] = temp;
            }

            if (TempG[j] < TempG[i])
            {
                temp = TempG[i];
                TempG[i] = TempG[j];
                TempG[j] = temp;
            }

            if (TempR[j] < TempR[i])
            {
                temp = TempR[i];
                TempR[i] = TempR[j];
                TempR[j] = temp;
            }
        }
    }

    if (filterSize % 2 != 0)  // Если фильтр нечетный
    {
        deviceData[yWx + 0] = TempB[(filterSize * filterSize - 1) / 2] * 255;  // Берем центральный элемент
        deviceData[yWx + 1] = TempG[(filterSize * filterSize - 1) / 2] * 255;  // Берем центральный элемент
        deviceData[yWx + 2] = TempR[(filterSize * filterSize - 1) / 2] * 255;  // Берем центральный элемент
        deviceData[yWx + 3] = 0;
    }
    else  // Если фильтр четный
    {
        deviceData[yWx + 0] = ((TempB[(filterSize * filterSize - 2) / 2] + TempB[(filterSize * filterSize) / 2]) / 2) * 255;  // Берем центральный элемент
        deviceData[yWx + 1] = ((TempG[(filterSize * filterSize - 2) / 2] + TempG[(filterSize * filterSize) / 2]) / 2) * 255;  // Берем центральный элемент
        deviceData[yWx + 2] = ((TempR[(filterSize * filterSize - 2) / 2] + TempR[(filterSize * filterSize) / 2]) / 2) * 255;  // Берем центральный элемент
        deviceData[yWx + 3] = 0;
    }
}

// Функция для вывода времени в зависимости от колличества нулей
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

// Функция для красивого оформления
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
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);  // Отключаем логи от openCV (Оставляем только ошибки)

    start_time = system_clock::now(); // Начальное времени программы

    Mat inputImage = imread(inputPath);  // Создаем и считываем изображение
    cvtColor(inputImage, inputImage, COLOR_BGR2BGRA);  // Выбираем цветокорекцию
    int imgWidth = inputImage.cols;  // Ширина изображения
    int imgHeight = inputImage.rows;  // Высота изображения
    int channels = inputImage.channels();  // Количество каналов B-blue, G-green, R-red, A-alpha

    end_time_input = system_clock::now(); // Конец времени загрузки

    Mat outputImage = Mat::zeros(imgHeight, imgWidth, CV_8UC4);  // Создаем выходное изображение

    cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar4>();  // Описывает формат значения возвращаемого из текстуры

    cudaArray* cuArray1;  // Объявляем массив куда
    uchar* deviceData = NULL;  // Переменная для обработки данных на device

    cudaMalloc(&deviceData, imgHeight * imgWidth * sizeof(uchar) * channels);  // Выделяем память на device
    cudaMallocArray(&cuArray1, &cuDesc, imgWidth, imgHeight);  // Выделяем место под массив на device
    
    cudaBindTextureToArray(&cudaTexture, cuArray1, &cuDesc);  // Привязываем текстуру к массиву

    cudaMemcpyToArray(cuArray1, 0, 0, inputImage.data, imgWidth * imgHeight * sizeof(uchar) * channels, cudaMemcpyHostToDevice);  // Копируем данные в массив

    end_time_load_memory = system_clock::now(); // Конец времени выделения памяти

    // Запуск kernel
    dim3 block(8, 8);  // Размер блока
    dim3 grid((imgWidth + block.x - 1) / block.x, (imgHeight + block.y - 1) / block.y);  // Размер решётки
    meanFilter << < grid, block, 0 >> > (deviceData, imgHeight, imgWidth, channels, filterSize);
    cudaThreadSynchronize();  // Синхронизируем потоки

    end_time_filter = system_clock::now(); // Конец времени загрузки

    cudaMemcpy(outputImage.data, deviceData, imgWidth * imgHeight * sizeof(uchar) * channels, cudaMemcpyDeviceToHost);  // Возвращаем данные с device

    imwrite(outputPath, outputImage);  // Выгружаем изображение

    end_time_output = system_clock::now(); // Конец выгрузки изображения

    // Освобождение памяти
    cudaUnbindTexture(cudaTexture);
    cudaFree(deviceData);
    cudaFree(cuArray1);

    end_time = system_clock::now(); // Конечное время программы

    if (timeText)
    {
        // Вывод времени выполнения работы
        cout << outputTime("Total program running time: ", start_time, end_time) << endl;
        cout << outlineTime("Total program running time: ", start_time, end_time) << endl;
        cout << outputTime("Image loading time: ", start_time, end_time_input) << endl;
        cout << outputTime("Memory allocation time: ", end_time_input, end_time_load_memory) << endl;
        //cout << outputTime("Data preparation time: ", end_time_load_memory, end_time_data) << endl;
        cout << outputTime("Filter application time: ", end_time_load_memory, end_time_filter) << endl;
        cout << outputTime("Image uploading time: ", end_time_filter, end_time_output) << endl;
        cout << outputTime("Memory release time: ", end_time_output, end_time) << endl;
        cout << outlineTime("Total program running time: ", start_time, end_time) << endl;
    }

    return 0;
}