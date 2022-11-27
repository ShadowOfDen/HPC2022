#include "main.h"

int main()
{
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);  // ��������� ���� �� openCV (��������� ������ ������)

    string fileName = "";

    vector<double> data_ch1;  // ������ �������� ������ ������
    vector<double> data_ch2;  // ������ �������� ������� ������

    wav_header_t header;  // ������� ������ ���������
    int samples_count = 0;  // ����������� ������ � data

    cout << "Input the path to the file: "; cin >> fileName;

    // ����� ���������� ����
    system_clock::time_point start_time, end_time_read, end_time_cpu, end_time_gpu, end_time;

    start_time = system_clock::now(); // ������ ������� GPU

    wav_Reader(header, fileName, samples_count, data_ch1, data_ch2);  // �������� ������ �� �����

    end_time_read = system_clock::now(); // ������ ������� GPU
    cout << outputTime("Data read execution time: ", start_time, end_time_read) << endl;

    // �������� ������
    if (processor == 0)
    {
        spectrogram_from_signal(header, samples_count, data_ch1, data_ch2);  // CPU

        end_time_cpu = system_clock::now();
        cout << outputTime("Data read execution time: ", end_time_read, end_time_cpu) << endl;
    }
    else if (processor == 1)
    {
        spectrogram_from_signal_cuda(header, samples_count, data_ch1, data_ch2);  // GPU

        end_time_gpu = system_clock::now();
        cout << outputTime("Data read execution time: ", end_time_read, end_time_gpu) << endl;
    }
    else
    {
        spectrogram_from_signal(header, samples_count, data_ch1, data_ch2);  // CPU

        end_time_cpu = system_clock::now();
        cout << outputTime("Data read execution time: ", end_time_read, end_time_cpu) << endl;

        spectrogram_from_signal_cuda(header, samples_count, data_ch1, data_ch2);  // GPU

        end_time_gpu = system_clock::now();
        cout << outputTime("Data read execution time: ", end_time_cpu, end_time_gpu) << endl;
    }

    cout << outputTime("Program execution time: ", start_time, end_time) << endl;

    waitKey();  // ������� ������� ������ �� ��������

    return 0;
}