#include "main.h"

void viewGraph(CvPlot::Axes& axes, Mat& mat, vector<double>& in, vector<double>& out, string color, string Name, int ml)
{
    string temp = "";

    axes.create<CvPlot::Series>(in, out, color);  // Отрисывываем левый канал
    mat = axes.render(height, width);  // Создание изображения для вывода

    if (ml == 1)
    {
        switch (mode)
        {
        case 1:

            if (interactive == 1)
                CvPlot::show(Name, axes);  // Отображаем график
            else
                imshow(Name, mat);  // Вывод изображения на экран

            break;

        case 2:

            temp = Name + extension;  // Добавим расширение для сохранения
            imwrite(temp, mat);  // Сохраняем изображение

            break;

        case 3:

            if (interactive == 1)
                CvPlot::show(Name, axes);  // Отображаем график
            else
                imshow(Name, mat);  // Вывод изображения на экран

            temp = Name + extension;  // Добавим расширение для сохранения
            imwrite(temp, mat);  // Сохраняем изображение

            break;
        }
    }
}

void viewGraph(wav_header_t& header, vector<double>& in, vector<double>& out1, vector<double>& out2, string Name, int modeP)
{
    CvPlot::Axes axes = CvPlot::makePlotAxes();  // Создаем поверхность
    axes = CvPlot::makePlotAxes();  // Создаем поверхность
    Mat mat;  // Поверхность для вывода

    string name = "";  // Переменная для хранения названия файла

    // Добавляем к названию файла приписку о процессоре
    if (modeP == 1) name = Name + "_CPU";
    else if (modeP == 2) name = Name + "_GPU";
    else name = Name;

    if (header.numChannels == 1)
    {
        viewGraph(axes, mat, in, out1, "-r", name, 1);  // Отрисовываем одно изображение
    }
    else
    {
        if (multichannel == 0)  // На разных изображениях
        {
            CvPlot::Axes axes2 = CvPlot::makePlotAxes();  // Создаем поверхность
            axes2 = CvPlot::makePlotAxes();  // Создаем поверхность
            Mat mat2;  // Поверхность для вывода

            string temp = name + "_1";
            viewGraph(axes, mat, in, out1, "-r", temp, 1);  // Отрисовываем первое изображение

            temp = name + "_2";
            viewGraph(axes2, mat2, in, out2, "-g", temp, 1);  // Отрисовываем второе изображение
        }
        else  // На одинаковом
        {
            viewGraph(axes, mat, in, out1, "-r", name, 0);  // Отрисовываем первый график
            viewGraph(axes, mat, in, out2, "-g", name, 1);  // Отрисовываем второй график
        }
    }
}
