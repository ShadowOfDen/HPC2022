#include "main.h"

void wav_Reader(wav_header_t& header, string name, int& samples_count, vector<double>& out1, vector<double>& out2)
{
    if (name[name.length() - 4] == '.' &&
        name[name.length() - 3] == 'w' &&
        name[name.length() - 2] == 'a' &&
        name[name.length() - 1] == 'v')
    {
        const char* charFileName = name.c_str();  // Преобразуем string в const char

        FILE* f;  // Переменная для файла
        errno_t error;  // Переменная для отлова ошибок
        if (error = fopen_s(&f, charFileName, "rb") == NULL)  // Если файл существует, то открываем его
        {
            fread(&header, sizeof(header), 1, f);  // Считываем данные файла в структуру

            if (*(unsigned long*)&header.chunkID == 0x46464952 &&
                *(unsigned long*)&header.format == 0x45564157 &&
                *(unsigned long*)&header.subChunk1ID == 0x20746d66)  // Проверяем wav файл на нужную структуру (RIFF, WAVE, fmt )
            {
                fseek(f, header.subChunk1Size - 16, SEEK_CUR);  //Пропустить дополнительные байты формата и дополнительные байты формата

                chunk_t chunk;  // Создаем объект структуры чанка для данных
                int k = 0;  // Попытки для счета data

                while (k < 10)  //Переходим к этим данным
                {
                    fread(&chunk, sizeof(chunk), 1, f);  // Читаем данные даты
                    if (*(unsigned long*)&chunk.subChunk2ID == 0x61746164) break;  // Если содержится слова дата, то выходим из цикла
                    fseek(f, chunk.subChunk2Size, SEEK_CUR); // иначе ищем дату дальше
                    k++;  // Увеличиваем попытку
                }

                int sample_size = header.bitsPerSample / 8;  // Число данных в data
                samples_count = chunk.subChunk2Size * 8 / header.bitsPerSample;  // Размер массива data

                unsigned long* data = new unsigned long[samples_count];  // Создаем динамический массив под data
                memset(data, 0, sizeof(unsigned long) * samples_count);  // Заполняем изначальное простарнсво в памяти нулями

                for (int i = 0; i < samples_count; i++)
                {
                    fread(&data[i], sample_size, 1, f);  // Считываем данные data
                }

                vector<double> x;  // Вектор значений x

                if (header.numChannels == 1)  // Если канал один
                {
                    for (int i = 0, k = 0; i < samples_count; i++, k++)
                    {
                        //for 16 bits per sample only
                        double step_1 = ((double)data[i] - 0x8000) / 0x8000;  // Переводим значение из 16-ричной системы
                        double step_2 = step_1 > 0.0 ? step_1 - 1.0 : step_1 + 1.0;  // Нормализируем
                        double step_3 = step_2 * 32760;  // Возвращаем по максимальной амплитуде

                        out1.push_back(step_3);  // Записываем значение в вектор

                        x.push_back((double)(k) / (double)(header.sampleRate));  // Значение по оси Х (секунды)
                    }

                    if (output == 0) viewGraph(header, x, out1, out2, "SoundGraph", 0);  // Отрисовываем графики
                }
                else  // Если канала два
                {

                    for (int i = 0, k = 0; i < samples_count - 1; i++, k++)
                    {
                        double step_1 = ((double)data[i] - 0x8000) / 0x8000;  // Переводим значение из 16-ричной системы
                        double step_2 = step_1 > 0.0 ? step_1 - 1.0 : step_1 + 1.0;  // Нормализируем
                        double step_3 = step_2 * 32760;  // Возвращаем по максимальной амплитуде

                        out1.push_back(step_3);

                        step_1 = ((double)data[i + 1] - 0x8000) / 0x8000;  // Переводим значение из 16-ричной системы
                        step_2 = step_1 > 0.0 ? step_1 - 1.0 : step_1 + 1.0;  // Нормализируем
                        step_3 = step_2 * 32760;  // Возвращаем по максимальной амплитуде

                        out2.push_back(step_3);

                        x.push_back((double)(k) / (double)(header.sampleRate));  // Значение по оси Х (секунды)
                        i++;
                    }

                    if (output == 0) viewGraph(header, x, out1, out2, "SoundGraph", 0);  // Отрисовываем графики

                }

                delete[] data;  // Очищаем память
            }
            else
            {
                cout << "Error 03: The file contains an incorrect structure!!!" << endl;
            }

            fclose(f);  // Закрываем файл
        }
        else
        {
            cout << "Error 02: File not found or file does not exist!!!" << endl;
        }
    }
    else
    {
        cout << "Error 01: The file has no extension .wav!!!" << endl;
    }
}
