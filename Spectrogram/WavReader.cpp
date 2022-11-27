#include "main.h"

void wav_Reader(wav_header_t& header, string name, int& samples_count, vector<double>& out1, vector<double>& out2)
{
    if (name[name.length() - 4] == '.' &&
        name[name.length() - 3] == 'w' &&
        name[name.length() - 2] == 'a' &&
        name[name.length() - 1] == 'v')
    {
        const char* charFileName = name.c_str();  // ����������� string � const char

        FILE* f;  // ���������� ��� �����
        errno_t error;  // ���������� ��� ������ ������
        if (error = fopen_s(&f, charFileName, "rb") == NULL)  // ���� ���� ����������, �� ��������� ���
        {
            fread(&header, sizeof(header), 1, f);  // ��������� ������ ����� � ���������

            if (*(unsigned long*)&header.chunkID == 0x46464952 &&
                *(unsigned long*)&header.format == 0x45564157 &&
                *(unsigned long*)&header.subChunk1ID == 0x20746d66)  // ��������� wav ���� �� ������ ��������� (RIFF, WAVE, fmt )
            {
                fseek(f, header.subChunk1Size - 16, SEEK_CUR);  //���������� �������������� ����� ������� � �������������� ����� �������

                chunk_t chunk;  // ������� ������ ��������� ����� ��� ������
                int k = 0;  // ������� ��� ����� data

                while (k < 10)  //��������� � ���� ������
                {
                    fread(&chunk, sizeof(chunk), 1, f);  // ������ ������ ����
                    if (*(unsigned long*)&chunk.subChunk2ID == 0x61746164) break;  // ���� ���������� ����� ����, �� ������� �� �����
                    fseek(f, chunk.subChunk2Size, SEEK_CUR); // ����� ���� ���� ������
                    k++;  // ����������� �������
                }

                int sample_size = header.bitsPerSample / 8;  // ����� ������ � data
                samples_count = chunk.subChunk2Size * 8 / header.bitsPerSample;  // ������ ������� data

                unsigned long* data = new unsigned long[samples_count];  // ������� ������������ ������ ��� data
                memset(data, 0, sizeof(unsigned long) * samples_count);  // ��������� ����������� ����������� � ������ ������

                for (int i = 0; i < samples_count; i++)
                {
                    fread(&data[i], sample_size, 1, f);  // ��������� ������ data
                }

                vector<double> x;  // ������ �������� x

                if (header.numChannels == 1)  // ���� ����� ����
                {
                    for (int i = 0, k = 0; i < samples_count; i++, k++)
                    {
                        //for 16 bits per sample only
                        double step_1 = ((double)data[i] - 0x8000) / 0x8000;  // ��������� �������� �� 16-������ �������
                        double step_2 = step_1 > 0.0 ? step_1 - 1.0 : step_1 + 1.0;  // �������������
                        double step_3 = step_2 * 32760;  // ���������� �� ������������ ���������

                        out1.push_back(step_3);  // ���������� �������� � ������

                        x.push_back((double)(k) / (double)(header.sampleRate));  // �������� �� ��� � (�������)
                    }

                    if (output == 0) viewGraph(header, x, out1, out2, "SoundGraph", 0);  // ������������ �������
                }
                else  // ���� ������ ���
                {

                    for (int i = 0, k = 0; i < samples_count - 1; i++, k++)
                    {
                        double step_1 = ((double)data[i] - 0x8000) / 0x8000;  // ��������� �������� �� 16-������ �������
                        double step_2 = step_1 > 0.0 ? step_1 - 1.0 : step_1 + 1.0;  // �������������
                        double step_3 = step_2 * 32760;  // ���������� �� ������������ ���������

                        out1.push_back(step_3);

                        step_1 = ((double)data[i + 1] - 0x8000) / 0x8000;  // ��������� �������� �� 16-������ �������
                        step_2 = step_1 > 0.0 ? step_1 - 1.0 : step_1 + 1.0;  // �������������
                        step_3 = step_2 * 32760;  // ���������� �� ������������ ���������

                        out2.push_back(step_3);

                        x.push_back((double)(k) / (double)(header.sampleRate));  // �������� �� ��� � (�������)
                        i++;
                    }

                    if (output == 0) viewGraph(header, x, out1, out2, "SoundGraph", 0);  // ������������ �������

                }

                delete[] data;  // ������� ������
            }
            else
            {
                cout << "Error 03: The file contains an incorrect structure!!!" << endl;
            }

            fclose(f);  // ��������� ����
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