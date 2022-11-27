#include "main.h"

int main()
{
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);  // Отключаем логи от openCV (Оставляем только ошибки)

    string fileName = "";

    vector<double> data_ch1;  // Вектор значений Левого канала
    vector<double> data_ch2;  // Вектор значений Правого канала

    wav_header_t header;  // Создаем объект структуры
    int samples_count = 0;  // Колличества данных в data

    cout << "Input the path to the file: "; cin >> fileName;

    // Время выполнения кода
    system_clock::time_point start_time, end_time_read, end_time_cpu, end_time_gpu, end_time;

    start_time = system_clock::now(); // Началь времени GPU

    wav_Reader(header, fileName, samples_count, data_ch1, data_ch2);  // Получаем данные со звука

    end_time_read = system_clock::now(); // Началь времени GPU
    cout << outputTime("Data read execution time: ", start_time, end_time_read) << endl;

    // Получаем спектр
    if (processor == 0)
    {
        spectrogram_from_signal(header, samples_count, data_ch1, data_ch2);  // CPU

        end_time_cpu = system_clock::now();
        cout << outputTime("The execution time of the Fourier transform CPU: ", end_time_read, end_time_cpu) << endl;
    }
    else if (processor == 1)
    {
        spectrogram_from_signal_cuda(header, samples_count, data_ch1, data_ch2);  // GPU

        end_time_gpu = system_clock::now();
        cout << outputTime("The execution time of the Fourier transform GPU: ", end_time_read, end_time_gpu) << endl;
    }
    else
    {
        spectrogram_from_signal(header, samples_count, data_ch1, data_ch2);  // CPU

        end_time_cpu = system_clock::now();
        cout << outputTime("The execution time of the Fourier transform CPU: ", end_time_read, end_time_cpu) << endl;

        spectrogram_from_signal_cuda(header, samples_count, data_ch1, data_ch2);  // GPU

        end_time_gpu = system_clock::now();
        cout << outputTime("The execution time of the Fourier transform GPU: ", end_time_cpu, end_time_gpu) << endl;
    }

    end_time = system_clock::now();
    cout << outputTime("Program execution time: ", start_time, end_time) << endl;

    waitKey();  // Ожидаем нажатия клавиш на графиках

    return 0;
}
