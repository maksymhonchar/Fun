#include "utils.h"

void openPort(HANDLE *hSerial)
{
    fprintf(stderr, "Opening serial port...\n");
    hSerial = CreateFile(
                  "\\\\.\\COM3", GENERIC_READ|GENERIC_WRITE, 0, NULL,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL );
    if (hSerial == INVALID_HANDLE_VALUE)
    {
        fprintf(stderr, "Error by opening a serial port.\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        fprintf(stderr, "OK\n");
    }
}

void setDeviceParameters(DCB *dcbSerialParams, HANDLE *hSerial)
{
    fprintf(stderr, "Setting device parameters...\n");
    dcbSerialParams->DCBlength = sizeof(*dcbSerialParams);
    if(GetCommState(*hSerial, dcbSerialParams) == 0)
    {
        fprintf(stderr, "Error getting device state.\n");
        CloseHandle(*hSerial);
        exit(EXIT_FAILURE);
    }
    dcbSerialParams->BaudRate = CBR_38400;
    dcbSerialParams->ByteSize = 8;
    dcbSerialParams->StopBits = ONESTOPBIT;
    dcbSerialParams->Parity = NOPARITY;
    if(SetCommState(*hSerial, dcbSerialParams) == 0)
    {
        fprintf(stderr, "Error setting device parameters.\n");
        CloseHandle(hSerial);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "OK\n");
}

void setTimeouts(COMMTIMEOUTS *timeouts, HANDLE *hSerial)
{
    timeouts->ReadIntervalTimeout = 50;
    timeouts->ReadTotalTimeoutConstant = 50;
    timeouts->ReadTotalTimeoutMultiplier = 10;
    timeouts->WriteTotalTimeoutConstant = 50;
    timeouts->WriteTotalTimeoutMultiplier = 10;
    if(SetCommTimeouts(*hSerial, timeouts) == 0)
    {
        fprintf(stderr, "Error setting timeouts.\n");
        CloseHandle(hSerial);
        exit(EXIT_FAILURE);
    }
}

void closePort(HANDLE *hSerial)
{
    fprintf(stderr, "Closing a serial port...\n");
    if(CloseHandle(*hSerial) == 0)
    {
        fprintf(stderr, "Error by closing a serial port.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Serial port closed correctly.");
}
