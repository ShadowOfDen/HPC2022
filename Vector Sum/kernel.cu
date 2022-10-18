#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace chrono;

//-------------Parametrs-------------

bool RorF = false; // Заполнение матриц рандомными числами или через файл (Random/File - False/True)
bool IorD = false; // Заполнение матриц рандомными числами int или double (Int/Double - False/True)
bool allText = false; // Включение/отключение всего выводимого текста (кроме время выполнения работы (Off/On - False/True)
bool inputText = true; // Включение/отключение входных матриц (Off/On - False/True)
bool outputText = true; // Включение/отключение выходной матрицы (Off/On - False/True)
int mode = 0; // Режим работы (0 - CPU, 1 - GPU, 2 - all)
bool check = false; // Режим проверки для работы GPU (Off/On - False/True)
int range[2] = { -100, 100 }; // Диапазон рандомных чисел, если не нужен заполнить нулями ( { -10, 10 } / { 0, 0 } )

int vector_size = 16777216; // Размер векторов

// Потоков в блоке, должно быть меньше, чем число N
// N / block_size без остатка!
int block_size = 256; // Потоков в блоке

size_t bytes; // Переменная хранящие размер матриц

// Матрицы host и device
double * h_A, * h_B, * h_C, * d_A, * d_B, * d_C;

system_clock::time_point start_time, end_time, end_time_input, end_time_input_display, start_time_output,
                          end_time_output, end_time_output_display, end_time_calculations, end_time_input_GPU,
                           start_time_GPU, end_time_calculations_GPU, end_time_output_1_GPU, end_time_verification,
                            end_time_output_2_GPU;

//-------------Functions-------------

__global__ void vec_add(double* A, double* B, double* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void init_Matrix(double* arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (!IorD) // Если флаг на целые числа
        {
            if (range[0] == 0 && range[1] == 0) // Если range пустой, генерируем числа без диапазона
            {
                if (rand() % 2 == 0) arr[i] = rand(); // Генерируем как положительные
                else arr[i] = -rand(); // Так и отрицательные числа
            }
            else arr[i] = rand() % (2 * range[1]) + range[0]; // Если range заполнен, генерируем по диапазону
        }
        else // Если флаг на дробные числа
        {
            if (range[0] == 0 && range[1] == 0) // Если range пустой, генерируем числа без диапазона
            {
                if (rand() % 2 == 0) arr[i] = (double)rand() / 1000;
                else arr[i] = -(double)rand() / 10000;
            }
            else // Если range заполнен, генерируем по диапазону
            {
                double temp = (double)rand() / RAND_MAX;
                arr[i] = range[0] + temp * (range[1] - range[0]);
            }
        }
    }
}

void verify_result(double* A, double* B, double* C, int size)
{
    for (int i = 0; i < size; i++) 
        assert(C[i] == A[i] + B[i]); // Проверяем значение на правильность
}

void memoryAllocation()
{
    bytes = sizeof(double) * vector_size; // Узнаем необходимый размер данных

    // Выделяем входные матрицы h_A и h_B в памяти хоста
    h_A = (double*)malloc(bytes);
    h_B = (double*)malloc(bytes);
    h_C = (double*)malloc(bytes);
}

void inputMass(double* arr, int size, istream& stream)
{
    string temp;
    for (int i = 0; i < size; i++)
    {
        stream >> temp;
        arr[i] = atof(temp.c_str());
    }
}

void outputMass(double* arr, int size, string title, ostream& stream)
{
    stream << title << endl;
    for (int i = 0; i < size; i++)
        stream << arr[i] << " ";
    stream << endl;
}

void vectorAddCPU(double* A, double* B, double* C, int size)
{
    for (int i = 0; i < size; i++)
        C[i] = A[i] + B[i]; // Считаем необходимое значение по циклу

    end_time_calculations = system_clock::now(); // Время после расчета
}

void vectorAddGPU(double* h_A, double* h_B, double* h_C, double* d_A, double* d_B, double* d_C, int size, size_t bytes, int blockSize)
{
    start_time_GPU = system_clock::now(); // Началь времени GPU

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = blockSize; // потоков в блоке
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock; // блоков в решетке

    end_time_input_GPU = system_clock::now(); // Время конца инициализации

    vec_add << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, size);

    end_time_calculations_GPU = system_clock::now(); // Время конца расчета

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    end_time_output_1_GPU = system_clock::now();

    verify_result(h_A, h_B, h_C, size);

    end_time_verification = system_clock::now();

    cudaFree(d_A);
    cudaFree(d_A);
    cudaFree(d_A);

    end_time_output_2_GPU = system_clock::now();
}

//-----------------------------------

int main()
{
    start_time = system_clock::now(); // Начальное времени программы

    if (!RorF) // Рандомные значения
    {
        srand(time(0)); // Рандомизация по времени

        memoryAllocation(); // Создание и выделение памяти

        // Инииализаия матриц
        init_Matrix(h_A, vector_size);
        init_Matrix(h_B, vector_size);
    }
    else // Значения из файла
    {
        ifstream fin;
        fin.open("input.txt");

        string temp;
        fin >> temp;
        vector_size = atoi(temp.c_str());

        cout << vector_size << endl;

        memoryAllocation(); // Создание и выделение памяти

        inputMass(h_A, vector_size, fin); // Считываем матрицы
        inputMass(h_B, vector_size, fin); // Считываем матрицы

        fin.close();
    }

    end_time_input = system_clock::now(); // Время после инициализации данных

    if (allText && inputText) // Если флаги включены, отображаем текст
    {
        outputMass(h_A, vector_size, "<<Vector A>>", cout);
        outputMass(h_B, vector_size, "<<Vector B>>", cout);
    }

    end_time_input_display = system_clock::now(); // Время после вывода данных

    if (mode == 0) // Если CPU
    {
        vectorAddCPU(h_A, h_B, h_C, vector_size);
    }
    else if (mode == 1) // Если GPU
    {
        vectorAddGPU(h_A, h_B, h_C, d_A, d_B, d_C, vector_size, bytes, block_size);
    }
    else // Все вместе
    {
        vectorAddCPU(h_A, h_B, h_C, vector_size);

        vectorAddGPU(h_A, h_B, h_C, d_A, d_B, d_C, vector_size, bytes, block_size);
    }

    start_time_output = system_clock::now();

    if (RorF)
    {
        ofstream fout;
        fout.open("output.txt");

        outputMass(h_C, vector_size, "<<Vector C>>", fout);

        fout.close();
    }

    end_time_output = system_clock::now(); // Время после выгрузки данных в файл

    if (allText && outputText) // Если флаги включены, отображаем текст
        outputMass(h_C, vector_size, "<<Vector C>>", cout);

    end_time_output_display = system_clock::now(); // Время после отображения данных

    free(h_A);
    free(h_B);
    free(h_C);

    end_time = system_clock::now(); // Конечное времени программы

    // Отображение времени работы программы

    cout << "Total program running time: " << duration_cast<nanoseconds>(end_time - start_time).count() << " ns" << endl;

    if (mode == 0)
    {
        cout << "Calculation time (CPU): " << duration_cast<nanoseconds>(end_time_calculations - end_time_input_display).count() << " ns" << endl;
        cout << "Data loading time: " << duration_cast<nanoseconds>(end_time_input - start_time).count() << " ns" << endl;
    }
    else if (mode == 1)
    {
        cout << "Calculation time (GPU): " << duration_cast<nanoseconds>(end_time_calculations_GPU - end_time_input_GPU).count() << " ns" << endl;

        if (check)
            cout << "Verification data time: " << duration_cast<nanoseconds>(end_time_verification - end_time_output_1_GPU).count() << " ns" << endl;

        cout << "Data loading time: " << duration_cast<nanoseconds>(end_time_input - start_time).count() + 
            duration_cast<nanoseconds>(end_time_input_GPU - start_time_GPU).count() << " ns" << endl;
    }
    else
    {
        cout << "Calculation time (CPU): " << duration_cast<nanoseconds>(end_time_calculations - end_time_input_display).count() << " ns" << endl;

        cout << "Calculation time (GPU): " << duration_cast<nanoseconds>(end_time_calculations_GPU - end_time_input_GPU).count() << " ns" << endl;

        if (check)
            cout << "Verification data time: " << duration_cast<nanoseconds>(end_time_verification - end_time_output_1_GPU).count() << " ns" << endl;

        cout << "Data loading time: " << duration_cast<nanoseconds>(end_time_input - start_time).count() +
            duration_cast<nanoseconds>(end_time_input_GPU - start_time_GPU).count() << " ns" << endl;

    }


    if (RorF) cout << "Data upload time: " << duration_cast<nanoseconds>(end_time_output - start_time_output).count() << " ns" << endl;
    if ((allText && inputText) || (allText && outputText)) cout << "Data Display time: " << duration_cast<nanoseconds>(end_time_input_display - end_time_input).count() +
        duration_cast<nanoseconds>(end_time_output_display - end_time_output).count() << " ns" << endl;

    return 0;
}