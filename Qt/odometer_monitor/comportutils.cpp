#include "comportutils.h"

#include <QDebug>

ComPortUtils::ComPortUtils()
{

}

QStringList ComPortUtils::getAvailablePortsNames()
{
    // Go through all available ports and return a list of their names.
    QStringList portsNamesList;
    QList<QSerialPortInfo> availablePortsList = QSerialPortInfo::availablePorts();
    foreach (QSerialPortInfo item, availablePortsList) {
        portsNamesList.push_back(item.portName());
    }
    return portsNamesList;
}

QString ComPortUtils::reverseHexToStr(QString rawHex, int bytesToRead)
{
    if(nullptr == rawHex || (rawHex.length() / HEX_DIGITS_IN_BYTE) != bytesToRead)
        return "";

    QString readyRawHex = "";
    int start = 0;
    for(int i = 0; i < bytesToRead; i++)
    {
        QString dataToPush = rawHex.mid(start, HEX_DIGITS_IN_BYTE);
        readyRawHex.push_front(dataToPush);
        start += HEX_DIGITS_IN_BYTE;
    }

    return readyRawHex;
}

int16_t ComPortUtils::processRawData(QString rawDataString)
{
    QString reversedHexVal = reverseHexToStr(rawDataString, HEX_DIGITS_IN_BYTE);
    return reversedHexVal.toInt(0, 16);
}
