#include "main.h"

void spectr(int size, vector<double>& data_channel, fftw_complex* in, fftw_complex* out, vector<double>& power)
{
    fftw_plan p;

    for (int i = 0; i < size; i++)
    {
        in[i][0] = data_channel[i];
        in[i][1] = (double)0;
    }

    p = fftw_plan_dft_1d(size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p);

    for (int i = 0; i < size; i++)
        power[i] = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);

    fftw_destroy_plan(p);
}

// Вычисление спектра периодического сигнала
void spectrogram_from_signal(wav_header_t& header, int samples_count, vector<double>& data_channel_1, vector<double>& data_channel_2)
{
    fftw_complex* in, * out;

    int size = 0;  // Задаем размер согласно количеству каналов

    if (header.numChannels == 1) size = samples_count;
    else size = samples_count / 2;

    in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);
    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * size);

    vector<double> power_ch1(size);  // Данные с первого канала
    vector<double> power_ch2(size);  // Данные со второго канала
    vector<double> frequency;  // Значения по x

    for (int i = 0; i < size; i++)
        frequency.push_back(((double)(header.sampleRate) / (double)(size)) * i);

    spectr(size, data_channel_1, in, out,power_ch1);
    spectr(size, data_channel_2, in, out,power_ch2);

    viewGraph(header, frequency, power_ch1, power_ch2, "Spectrogram", 1);

    fftw_cleanup();
}