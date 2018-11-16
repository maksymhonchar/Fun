#include "comport.h"

ComPort::ComPort(QString name)
{
    this->serialPort = new QSerialPort();
    this->portName = name;
}

bool ComPort::openPort()
{
    // Set port settings.
    this->setPortSettings(this->portName, 115200);

    // Open a connection.
    if(!this->serialPort->open(QIODevice::ReadOnly))
        return false;

    // Return success status of the connection.
    return true;
}

void ComPort::closePort()
{
    this->serialPort->close();
}

void ComPort::clearPortBuffer()
{
    this->serialPort->clear();
}

QString ComPort::readData(int maxSize)
{
    // Read maxSize bytes of data from the COM port.
    // Return hexademical interpretation of the received data.
    QByteArray bytesReceived = serialPort->read(maxSize);
    return bytesReceived.toHex();
}

QSerialPort *ComPort::getSerialPort()
{
    return this->serialPort;
}

void ComPort::setPortSettings(QString name, int baudRate)
{
    // Set the main COM port settings.
    this->serialPort->setPortName(name);
    this->serialPort->setBaudRate(baudRate);
    this->serialPort->setDataBits(QSerialPort::Data8);
    this->serialPort->setParity(QSerialPort::NoParity);
    this->serialPort->setStopBits(QSerialPort::OneStop);
    this->serialPort->setFlowControl(QSerialPort::NoFlowControl);
}
