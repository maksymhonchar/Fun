#ifndef COMPORTUTILS_H
#define COMPORTUTILS_H

#include <QtSerialPort/QSerialPortInfo>
#include <QStringList>
#include <QList>

#define HEX_DIGITS_IN_BYTE 2

class ComPortUtils
{
public:
    ComPortUtils();

    // Get a list of COM port names.
    QStringList getAvailablePortsNames();

    // Reverse hexademical number.
    QString reverseHexToStr(QString rawHex, int bytesToRead);

    // Process raw data from COMport.
    int16_t processRawData(QString rawDataString);

};

#endif // COMPORTUTILS_H
