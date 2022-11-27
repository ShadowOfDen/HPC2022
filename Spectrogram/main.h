#pragma once

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

// ����������� ����������
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

// ���������

// ������ �������� �����������
#define width 1280
#define height 720

// ���������� �������� �����������
#define extension ".bmp"

// ��������� �����������
#define mode 0  // ����� ����������� ������ (0 - �� ���������� � �� ���������, 1 - ���������� �����������, 2 - ��������� �����������, 3 - ���������� � ���������

#define interactive 0  // ����� ������������� �������� (0 - ���������, 1 - ��������)

#define multichannel 1  // ����� ����������������� �� ����� ����������� (0 - �� ������ ������������, 1 - �� ����� �����������)

#define output 0  // ����� ����������� (0 - ��� �����������, 1 - ������ �������)

#define processor 2  // ������������� ���������� (0 - CPU, 1 - GPU, 2 - CPU & GPU)

// ��������� ������ ��� wavReader
struct wav_header_t
{
    char chunkID[4];  // "RIFF" = 0x46464952
    unsigned long chunkSize;  // 36 + SubChunk2Size
    char format[4];  // "WAVE" = 0x45564157
    char subChunk1ID[4];  // "fmt " = 0x20746d66
    unsigned long subChunk1Size;  // ��� ������ ��������� ����� ����������, ������� ������� �� ���� ������
    unsigned short audioFormat;  // �������� audioFormat = 1 (�.�. �������� �����������), �������� �� 1, ��������� �� ��������� ����� ������.
    unsigned short numChannels;  // Mono = 1, Stereo = 2
    unsigned long sampleRate;  // 8000, 44100, � �.�.
    unsigned long byteRate;  // = SampleRate * NumChannels * BitsPerSample/8
    unsigned short blockAlign;  // = NumChannels * BitsPerSample/8
    unsigned short bitsPerSample;  // 8 bits = 8, 16 bits = 16, � �.�.
};

struct chunk_t
{
    char subChunk2ID[4]; //"data" = 0x61746164
    unsigned long subChunk2Size;  // = NumSamples * NumChannels * BitsPerSample/8
};


// ��������� ������� ��� ��������� ���������� ������
string outputTime(string title, system_clock::time_point start, system_clock::time_point end);
string outlineTime(string title, system_clock::time_point start, system_clock::time_point end);

// ��������� ��� ������ �����������
void viewGraph(wav_header_t& header, vector<double>& in, vector<double>& out1, vector<double>& out2, string Name, int modeP);

// �������� ������� ��� ���������� ���� �� wav
void wav_Reader(wav_header_t& header, string name, int& samples_count, vector<double>& out1, vector<double>& out2);

// �������� ������� ������������� �� CPU
void spectrogram_from_signal(wav_header_t& header, int samples_count, vector<double>& data_channel_1, vector<double>& data_channel_2);

// �������� ������� ������������� �� GPU
void spectrogram_from_signal_cuda(wav_header_t& header, int samples_count, vector<double>& data_channel_1, vector<double>& data_channel_2);

// �������� ������� ������ �������
string outputTime(string title, system_clock::time_point start, system_clock::time_point end);