#pragma once

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

// Стандартные библиотеки
#include <iostream>
#include <string.h>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <chrono>

// OpenCV
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

// MatPlot OpenCV
#define CVPLOT_HEADER_ONLY
#include <CvPlot/cvplot.h>
#include <CvPlot/core/Axes.h>

// FFTW
#include <fftw3.h>

// Namespace
using namespace cv;
using namespace std;
using namespace chrono;

// Параметры

// Размер выходных изображений
#define width 1280
#define height 720

// Расширение выходных изображений
#define extension ".bmp"

// Параметры отображения
#define mode 0  // Режим отображения данных (0 - не отображать и не сохранять, 1 - отображать изображения, 2 - сохранять изображения, 3 - отображать и сохранять

#define interactive 0  // Режим интерактивных графиков (0 - выключить, 1 - включить)

#define multichannel 1  // Режим мультикональности на одном изображении (0 - на разных изображениях, 1 - на одном изображении)

#define output 0  // Показ изображений (0 - все изображения, 1 - только спектры)

#define processor 2  // Использование процессора (0 - CPU, 1 - GPU, 2 - CPU & GPU)

// Структуры данных для wavReader
struct wav_header_t
{
    char chunkID[4];  // "RIFF" = 0x46464952
    unsigned long chunkSize;  // 36 + SubChunk2Size
    char format[4];  // "WAVE" = 0x45564157
    char subChunk1ID[4];  // "fmt " = 0x20746d66
    unsigned long subChunk1Size;  // Это размер остальной части подраздела, который следует за этим числом
    unsigned short audioFormat;  // Значения audioFormat = 1 (т.е. линейное квантование), отличные от 1, указывают на некоторую форму сжатия.
    unsigned short numChannels;  // Mono = 1, Stereo = 2
    unsigned long sampleRate;  // 8000, 44100, и т.д.
    unsigned long byteRate;  // = SampleRate * NumChannels * BitsPerSample/8
    unsigned short blockAlign;  // = NumChannels * BitsPerSample/8
    unsigned short bitsPerSample;  // 8 bits = 8, 16 bits = 16, и т.д.
};

struct chunk_t
{
    char subChunk2ID[4]; //"data" = 0x61746164
    unsigned long subChunk2Size;  // = NumSamples * NumChannels * BitsPerSample/8
};


// Прототипы функций для красивого оформления вывода
string outputTime(string title, system_clock::time_point start, system_clock::time_point end);
string outlineTime(string title, system_clock::time_point start, system_clock::time_point end);

// Прототипы для вывода изображений
void viewGraph(wav_header_t& header, vector<double>& in, vector<double>& out1, vector<double>& out2, string Name, int modeP);

// Прототип функции для считывания даты из wav
void wav_Reader(wav_header_t& header, string name, int& samples_count, vector<double>& out1, vector<double>& out2);

// Прототип функции спектрограммы на CPU
void spectrogram_from_signal(wav_header_t& header, int samples_count, vector<double>& data_channel_1, vector<double>& data_channel_2);

// Прототип функции спектрограммы на GPU
void spectrogram_from_signal_cuda(wav_header_t& header, int samples_count, vector<double>& data_channel_1, vector<double>& data_channel_2);

// Прототип функции вывода времени
string outputTime(string title, system_clock::time_point start, system_clock::time_point end);