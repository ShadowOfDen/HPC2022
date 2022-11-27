#include "main.h"

void viewGraph(CvPlot::Axes& axes, Mat& mat, vector<double>& in, vector<double>& out, string color, string Name, int ml)
{
    string temp = "";

    axes.create<CvPlot::Series>(in, out, color);  // ������������ ����� �����
    mat = axes.render(height, width);  // �������� ����������� ��� ������

    if (ml == 1)
    {
        switch (mode)
        {
        case 1:

            if (interactive == 1)
                CvPlot::show(Name, axes);  // ���������� ������
            else
                imshow(Name, mat);  // ����� ����������� �� �����

            break;

        case 2:

            temp = Name + extension;  // ������� ���������� ��� ����������
            imwrite(temp, mat);  // ��������� �����������

            break;

        case 3:

            if (interactive == 1)
                CvPlot::show(Name, axes);  // ���������� ������
            else
                imshow(Name, mat);  // ����� ����������� �� �����

            temp = Name + extension;  // ������� ���������� ��� ����������
            imwrite(temp, mat);  // ��������� �����������

            break;
        }
    }
}

void viewGraph(wav_header_t& header, vector<double>& in, vector<double>& out1, vector<double>& out2, string Name, int modeP)
{
    CvPlot::Axes axes = CvPlot::makePlotAxes();  // ������� �����������
    axes = CvPlot::makePlotAxes();  // ������� �����������
    Mat mat;  // ����������� ��� ������

    string name = "";  // ���������� ��� �������� �������� �����

    // ��������� � �������� ����� �������� � ����������
    if (modeP == 1) name = Name + "_CPU";
    else if (modeP == 2) name = Name + "_GPU";
    else name = Name;

    if (header.numChannels == 1)
    {
        viewGraph(axes, mat, in, out1, "-r", name, 1);  // ������������ ���� �����������
    }
    else
    {
        if (multichannel == 0)  // �� ������ ������������
        {
            CvPlot::Axes axes2 = CvPlot::makePlotAxes();  // ������� �����������
            axes2 = CvPlot::makePlotAxes();  // ������� �����������
            Mat mat2;  // ����������� ��� ������

            string temp = name + "_1";
            viewGraph(axes, mat, in, out1, "-r", temp, 1);  // ������������ ������ �����������

            temp = name + "_2";
            viewGraph(axes2, mat2, in, out2, "-g", temp, 1);  // ������������ ������ �����������
        }
        else  // �� ����������
        {
            viewGraph(axes, mat, in, out1, "-r", name, 0);  // ������������ ������ ������
            viewGraph(axes, mat, in, out2, "-g", name, 1);  // ������������ ������ ������
        }
    }
}