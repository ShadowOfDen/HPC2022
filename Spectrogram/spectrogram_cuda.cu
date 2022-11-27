#include "main.h"

#define BLOCK_SIZE 1024

__global__ void cufftMultiply(cufftComplex* idata, cufftComplex* odata, int size)
{
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < size)
    {
        odata[threadID].x = sqrt(idata[threadID].x * idata[threadID].x + idata[threadID].y * idata[threadID].y);  // Преобразовываем данные в мощность
    }
}


void spectr(int size, vector<double>& data_channel, cufftComplex* data_Host, cufftComplex* data_dev, vector<double>& power)
{
    cufftHandle plan;  // Создаем дескриптор cuFFT

    // Исходные данные
    for (int i = 0; i < size; i++)
    {
        data_Host[i].x = data_channel[i];
        data_Host[i].y = (double)0;
    }

    dim3 dimBlock(BLOCK_SIZE); // блок потока
    dim3 dimGrid((size + BLOCK_SIZE - 1) / dimBlock.x); // сетка потоков

    cufftPlan1d(&plan, size, CUFFT_C2C, 1);

    cudaMemset(data_dev, 0, sizeof(cufftComplex) * size);  // Первоначально заполняем 0
    cudaMemcpy(data_dev, data_Host, size * sizeof(cufftComplex), cudaMemcpyHostToDevice);  // Копируем из памяти хоста в память устройства

    cufftExecC2C(plan, data_dev, data_dev, CUFFT_FORWARD);  // Выполняем cuFFT, положительное преобразование
    cufftMultiply << <dimGrid, dimBlock >> > (data_dev, data_dev, size);
    cudaMemcpy(data_Host, data_dev, size * sizeof(cufftComplex), cudaMemcpyDeviceToHost);  // Копируем из памяти устройства в память хоста

    for (int i = 0; i < size; i++)
        //power[i] = sqrt(data_Host[i].x * data_Host[i].x + data_Host[i].y * data_Host[i].y);  // Преобразовываем данные в мощность
        power[i] = data_Host[i].x;  // Преобразовываем данные в мощность

    cufftDestroy(plan);  // Уничтожаем дескриптор
}

void spectrogram_from_signal_cuda(wav_header_t& header, int samples_count, vector<double>& data_channel_1, vector<double>& data_channel_2)
{
    int size = 0;  // Задаем размер согласно количеству каналов

    if (header.numChannels == 1) size = samples_count;
    else size = samples_count / 2;

    vector<double> power_ch1(size);  // Данные с первого канала
    vector<double> power_ch2(size);  // Данные со второго канала
    vector<double> frequency;  // Значения по x

    for (int i = 0; i < size; i++)
        frequency.push_back(((double)(header.sampleRate) / (double)(size)) * i);

    cufftComplex* data_dev; // Данные на стороне устройства
    cufftComplex* data_Host = (cufftComplex*)malloc(size * sizeof(cufftComplex));  // Данные на стороне хоста

    cudaMalloc((void**)&data_dev, sizeof(cufftComplex) * size);  // Выделяем память на устройстве

    spectr(size, data_channel_1, data_Host, data_dev, power_ch1);
    spectr(size, data_channel_2, data_Host, data_dev, power_ch2);

    viewGraph(header, frequency, power_ch1, power_ch2, "Spectrogram", 2);

    cudaFree(data_dev); // освободить место
    //cudaFree(data_Host); // освободить место
    free(data_Host);  // освободить место
}
