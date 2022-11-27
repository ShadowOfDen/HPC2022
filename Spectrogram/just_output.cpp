#include "main.h"

// Функция для вывода времени в зависимости от колличества нулей
string outputTime(string title, system_clock::time_point start, system_clock::time_point end)
{
    if (duration_cast<nanoseconds>(end - start).count() > 3600000000000)
    {
        title += to_string(duration_cast<nanoseconds>(end - start).count() / 3600000000000);
        title += ".";
        title += to_string(duration_cast<nanoseconds>(end - start).count() -
            (duration_cast<nanoseconds>(end - start).count() / 3600000000000) * 3600000000000);
        title += " hours";
    }
    else if (duration_cast<nanoseconds>(end - start).count() > 60000000000)
    {
        title += to_string(duration_cast<nanoseconds>(end - start).count() / 60000000000);
        title += ".";
        title += to_string(duration_cast<nanoseconds>(end - start).count() -
            (duration_cast<nanoseconds>(end - start).count() / 60000000000) * 60000000000);
        title += " min";
    }
    else if (duration_cast<nanoseconds>(end - start).count() > 1000000000)
    {
        title += to_string(duration_cast<nanoseconds>(end - start).count() / 1000000000);
        title += ".";
        title += to_string(duration_cast<nanoseconds>(end - start).count() -
            (duration_cast<nanoseconds>(end - start).count() / 1000000000) * 1000000000);
        title += " s";
    }
    else if (duration_cast<nanoseconds>(end - start).count() > 1000000)
    {
        title += to_string(duration_cast<nanoseconds>(end - start).count() / 1000000);
        title += ".";
        title += to_string(duration_cast<nanoseconds>(end - start).count() -
            (duration_cast<nanoseconds>(end - start).count() / 1000000) * 1000000);
        title += " ms";
    }
    else if (duration_cast<nanoseconds>(end - start).count() > 1000)
    {
        title += to_string(duration_cast<nanoseconds>(end - start).count() / 1000);
        title += ".";
        title += to_string(duration_cast<nanoseconds>(end - start).count() -
            (duration_cast<nanoseconds>(end - start).count() / 1000) * 1000);
        title += " us";
    }
    else
    {
        title += to_string(duration_cast<nanoseconds>(end - start).count());
        title += " ns";
    }

    return title;
}

// Функция для красивого оформления
string outlineTime(string title, system_clock::time_point start, system_clock::time_point end)
{
    string temp = "";

    title = outputTime(title, start, end);

    for (int i = 0; i < title.length(); i++)
        temp += "-";

    return temp;
}
