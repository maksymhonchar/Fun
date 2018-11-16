#ifndef COMPORT_H
#define COMPORT_H

#include <QtSerialPort/QSerialPort>


class ComPort
{
public:
    ComPort(QString name);

    // Opening and closing a COM port.
    bool openPort();
    void closePort();

    // Methods for operating the data related to COM port.
    void clearPortBuffer();
    QString readData(int maxSize);

    // Used only for connecting port signals to slots.
    QSerialPort *getSerialPort();

private:
    // Serial port instance to work with.
    QSerialPort *serialPort;
    QString portName;

    void setPortSettings(QString name, int baudRate);
};

#endif // COMPORT_H
