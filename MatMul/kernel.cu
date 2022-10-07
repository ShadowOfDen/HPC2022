#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

//-------------Parametrs-------------

bool RorF = false; // Заполнение матриц рандомными числами или через файл (Random/File - False/True)
bool IorD = false; // Заполнение матриц рандомными числами int или double (Int/Double - False/True)
bool allText = false; // Включение/отключение всего выводимого текста (кроме время выполнения работы (Off/On - False/True)
bool inputText = true; // Включение/отключение входных матриц (Off/On - False/True)
bool outputText = true; // Включение/отключение выходной матрицы (Off/On - False/True)
int mode = 2; // Режим работы (0 - CPU, 1 - GPU, 2 - all)
bool check = false; // Режим проверки для работы GPU (Off/On - False/True)
int range[2] = { -100, 100 }; // Диапазон рандомных чисел, если не нужен заполнить нулями ( { -10, 10 } / { 0, 0 } )

int sizeA[2] = { 8192, 8192 };  // Размер матрицы А
int sizeB[2] = { 8192, 8192 };  // Размер матрицы B
int sizeC[2];  // Размер матрицы C

// Потоков в блоке, должно быть меньше, чем число N
// N / block_size без остатка!
int block_size = 16; // Потоков в блоке

size_t size_A, size_B, size_C; // Переменные хранящие размер матриц


// Матрицы host и device
double* h_A, * h_B, * h_C, * d_A, * d_B, * d_C;

// Переменные времени
double start_time, end_time, end_time_input, end_time_input_display, 
          start_time_output, end_time_output_display, end_time_output, 
             end_time_calculations, start_time_GPU, end_time_input_GPU,
                 end_time_calculations_GPU, end_time_output_1_GPU, end_time_output_2_GPU, 
                    end_time_verification;

//-------------Functions-------------

__global__ void matrixMul(double*A, double*B, double*C, int N, int M)
{
    // Вычисляем строку каждого потока
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Вычисляем столбец каждого потока
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    C[row * M + column] = 0;

    for (int i = 0; i < N; i++)
        C[row * M + column] += A[row * N + i] * B[i * M + column];
}

void init_Matrix(double* arr, int rows, int columns)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            if (!IorD) // Если флаг на целые числа
            {
                if (range[0] == 0 && range[1] == 0) // Если range пустой, генерируем числа без диапазона
                {
                    if (rand() % 2 == 0) arr[i * columns + j] = rand(); // Генерируем как положительные
                    else arr[i * columns + j] = -rand(); // Так и отрицательные числа
                }
                else arr[i * columns + j] = rand() % (2 * range[1]) + range[0]; // Если range заполнен, генерируем по диапазону
            }
            else // Если флаг на дробные числа
            {
                if (range[0] == 0 && range[1] == 0) // Если range пустой, генерируем числа без диапазона
                {
                    if (rand() % 2 == 0) arr[i * columns + j] = (double)rand() / 1000;
                    else arr[i * columns + j] = -(double)rand() / 10000;
                }
                else // Если range заполнен, генерируем по диапазону
                {
                    double temp = (double)rand() / RAND_MAX;
                    arr[i * columns + j] = range[0] + temp * (range[1] - range[0]);
                }
            }
        }
    }
}

void verify_result(double* a, double* b, double* c, int C0, int C1, int A1)
{
    double temp = 0;

    for (int i = 0; i < C0; i++)
        for (int j = 0; j < C1; j++)
        {
            temp = 0;
            for (int k = 0; k < A1; k++)
                temp += a[i * A1 + k] * b[k * C1 + j];

            assert(c[i * C1 + j] == temp);
        }
}

void setSize(int* arr, istream& stream)
{
    string temp;

    for (int i = 0; i < 2; i++)
    {
        stream >> temp;
        arr[i] = atoi(temp.c_str());
    }
}

void inputMass(double* arr, int rows, int columns, istream& stream)
{
    string temp;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
        {
            stream >> temp;
            arr[i * columns + j] = atof(temp.c_str());
        }
    }
}

void outputMass(double* arr, int rows, int columns, string title, ostream& stream)
{
    stream << title << endl;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
            stream << arr[i * columns + j] << " ";
        stream << endl;
    }
    stream << endl;
}

void memoryAllocation()
{
    sizeC[0] = sizeA[0]; sizeC[1] = sizeB[1];

    size_A = sizeA[0] * sizeA[1] * sizeof(double); // Узнаем сколько памяти надо для матрицы А
    size_B = sizeB[0] * sizeB[1] * sizeof(double); // Узнаем сколько памяти надо для матрицы B
    size_C = sizeC[0] * sizeC[1] * sizeof(double); // Узнаем сколько памяти надо для матрицы C

    // Выделяем входные матрицы h_A и h_B в памяти хоста
    h_A = (double*)malloc(size_A);
    h_B = (double*)malloc(size_B);
    h_C = (double*)malloc(size_C);
}

void matrixMulCPU(double* A, double* B, double* C, int C0, int C1, int A1)
{
    for (int i = 0; i < C0; i++)
        for (int j = 0; j < C1; j++)
        {
            C[i * C1 + j] = 0;
            for (int k = 0; k < A1; k++)
                C[i * C1 + j] += A[i * A1 + k] * B[k * C1 + j];
        }

    end_time_calculations = clock(); // Время после расчета
}

void matrixMulGPU(double* A, double* B, double* C, int C0, int C1, int A0, int A1, int B1, size_t sizeA, size_t sizeB, size_t sizeC)
{
    start_time_GPU = clock(); // Началь времени GPU

    // Выделяем вектора в памяти устройства
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Копируем векторы из памяти хоста в память устройства
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // dim3 объекты
    dim3 threads(block_size, block_size);
    dim3 grid(B1 / threads.x, A0 / threads.y);

    end_time_input_GPU = clock(); // Время конца инициализации

    // Запуск kernel
    matrixMul << < grid, threads >> > (d_A, d_B, d_C, A1, B1);

    end_time_calculations_GPU = clock(); // Время конца расчета

    // Скопируем результат из памяти устройства в память хоста
    // h_C содержит результат в памяти хоста
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    end_time_output_1_GPU = clock();

    // Проверка результата
    if(check)
        verify_result(h_A, h_B, h_C, C0, C1, A1);

    end_time_verification = clock();

    // Отчитска памяти устройства
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    end_time_output_2_GPU = clock();
}

//-----------------------------------

int main()
{
    if (sizeA[1] == sizeB[0]) // Если размеры матриц не равны
    {
        start_time = clock(); // Начальное время

        if (!RorF) // Рандомные значения
        {
            srand(time(0)); // Рандомизация по времени

            memoryAllocation(); // Создание и выделение памяти

            // Инииализаия матриц
            init_Matrix(h_A, sizeA[0], sizeA[1]);
            init_Matrix(h_B, sizeB[0], sizeB[1]);
        }
        else // Значения из файла
        {
            ifstream fin;
            fin.open("input.txt");

            string temp;

            setSize(sizeA, fin);
            setSize(sizeB, fin);

            if (sizeA[1] != sizeB[0]) // Если считанные размеры дают ошибку
            {
                fin.close();
                goto stop;
            }

            memoryAllocation(); // Создание и выделение памяти

            inputMass(h_A, sizeA[0], sizeA[1], fin); // Считываем матрицы
            inputMass(h_B, sizeB[0], sizeB[1], fin); // Считываем матрицы

            fin.close();
        }

        end_time_input = clock(); // Время после инициализации данных

        if (allText && inputText) // Если флаги включены, отображаем текст
        {
            outputMass(h_A, sizeA[0], sizeA[1], "<<Matrix A>>", cout);
            outputMass(h_B, sizeB[0], sizeB[1], "<<Matrix B>>", cout);
        }

        end_time_input_display = clock(); // Время после вывода данных

        if (mode == 0) // Если CPU
        {
            matrixMulCPU(h_A, h_B, h_C, sizeC[0], sizeC[1], sizeA[1]);
        }
        else if(mode == 1) // Если GPU
        {
            matrixMulGPU(h_A, h_B, h_C, sizeC[0], sizeC[1], sizeA[0], sizeA[1], sizeB[1], size_A, size_B, size_C);
        }
        else // Все вместе
        {
            matrixMulCPU(h_A, h_B, h_C, sizeC[0], sizeC[1], sizeA[1]);

            matrixMulGPU(h_A, h_B, h_C, sizeC[0], sizeC[1], sizeA[0], sizeA[1], sizeB[1], size_A, size_B, size_C);
        }

        start_time_output = clock();

        if (RorF)
        {
            ofstream fout;
            fout.open("output.txt");

            outputMass(h_C, sizeC[0], sizeC[1], "<<Matrix C>>", fout);

            fout.close();
        }

        end_time_output = clock(); // Время после выгрузки данных в файл

        if (allText && outputText) // Если флаги включены, отображаем текст
            outputMass(h_C, sizeC[0], sizeC[1], "<<Matrix C>>", cout);

        end_time_output_display = clock(); // Время после отображения данных

        // Отчистка памяти хоста
        free(h_A);
        free(h_B);
        free(h_C);

        end_time = clock(); // Время конца программы

        // Отображение времени работы программы

        cout << "Total program running time: " << (double)(end_time - start_time) / CLOCKS_PER_SEC << " s" << endl;
        
        if (mode == 0)
        {
            cout << "Calculation time (CPU): " << (double)(end_time_calculations - end_time_input_display) / CLOCKS_PER_SEC << " s" << endl;
            cout << "Data loading time: " << (double)(end_time_input - start_time) / CLOCKS_PER_SEC << " s" << endl;
        }
        else if (mode == 1)
        {
            cout << "Calculation time (GPU): " << (double)(end_time_calculations_GPU - end_time_input_GPU) / CLOCKS_PER_SEC << " s" << endl;

            if (check)
                cout << "Verification data time: " << (double)(end_time_verification - end_time_output_1_GPU) / CLOCKS_PER_SEC << " s" << endl;

            cout << "Data loading time: " << (double)((end_time_input - start_time) +
                (end_time_input_GPU - start_time_GPU))/ CLOCKS_PER_SEC << " s" << endl;
        }
        else
        {
            cout << "Calculation time (CPU): " << (double)(end_time_calculations - end_time_input_display) / CLOCKS_PER_SEC << " s" << endl;

            cout << "Calculation time (GPU): " << (double)(end_time_calculations_GPU - end_time_input_GPU) / CLOCKS_PER_SEC << " s" << endl;

            if (check)
                cout << "Verification data time: " << (double)(end_time_verification - end_time_output_1_GPU) / CLOCKS_PER_SEC << " s" << endl;

            cout << "Data loading time: " << (double)((end_time_input - start_time) +
                (end_time_input_GPU - start_time_GPU)) / CLOCKS_PER_SEC << " s" << endl;

        }


        if (RorF) cout << "Data upload time: " << (double)(end_time_output - start_time_output) / CLOCKS_PER_SEC << " s" << endl;
        if ((allText && inputText) || (allText && outputText)) cout << "Data Display time: " << (double)((end_time_input_display - end_time_input) + (end_time_output_display - end_time_output)) / CLOCKS_PER_SEC << " s" << endl;
    }
    else
    {
    stop:
        cout << "Error: The number of columns of matrix A does not match the number of rows of matrix B!" << endl;
    }
}