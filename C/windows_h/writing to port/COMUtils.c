#include <stdio.h>
#include "COMUtils.h"


DWORD sendData(const char *data, DWORD size)
{
    /*
    DESCRIPTION:
        Sends bytes to the COM port.

    PARAMS:
        HANDLE *hCom - pointer to COM port handler.
        const char *data - data to send via COM port.
        DWORD size - size of data to send.

    RETURNS:
        Amount of bytes that was sent.
    */

    DWORD numberOfBytesWritten;
    WriteFile(hCom, data, size, &numberOfBytesWritten, 0);
    return (numberOfBytesWritten);
}

void printCommState(DCB *dcb)
{
    /*
    DESCRIPTION:
        Prints state of the serial communication device.

    PARAMS:
        DCB *dcb - pointer to the structure, that describes a serial communication device.
    */

    printf("COM port state: \n");
    printf("BaudRate = %u, ByteSize = %u, Parity = %u, StopBits = %u\n",
           dcb->BaudRate,
           dcb->ByteSize,
           dcb->Parity,
           dcb->StopBits );
}

int initCOM()
{
    /*
    DESCRIPTION:
        Initializes the COM port.

    RETURNS:
        Status code, see from the list above.
    */

    /*
    RETURN ERROR CODES:
    [0] - Success.
    [1] - CreateFile error.
    [2] - GetCommState error.
    [3] - SetCommState error.
    */

    // Status code to check success of the functions.
    BOOL fSuccess;

    // COM port address.
    const char *pcComPort = "\\\\.\\COM3";

    // Open a handle to the specified COM port.
    hCom = CreateFile( pcComPort,
                       GENERIC_READ | GENERIC_WRITE,
                       0,
                       NULL,
                       OPEN_EXISTING,
                       0,
                       NULL );
    if(hCom == INVALID_HANDLE_VALUE)
    {
        fprintf(stderr, "CreateFile failed with error %u.\n", GetLastError());
        return (1);
    }

    // Initialize DCB structure.
    ZeroMemory(&dcb, sizeof(DCB));
    dcb.DCBlength = sizeof(DCB);

    // Build the current configuration by first retrieving all current settings.
    fSuccess = GetCommState(hCom, &dcb);
    if(!fSuccess)
    {
        fprintf(stderr, "GetCommState failed with error %u.\n", GetLastError());
        return (2);
    }
    // Print state of the COM port before changing.
    printCommState(&dcb);

    // Fill in some DCB values and set the COM state.
    dcb.BaudRate = CBR_9600;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    fSuccess = SetCommState(hCom, &dcb);
    if(!fSuccess)
    {
        fprintf(stderr, "SetCommState failed with error %u.\n", GetLastError());
        return (3);
    }

    // Get the COM configuration again.
    fSuccess = GetCommState(hCom, &dcb);
    if(!fSuccess)
    {
        fprintf(stderr, "GetCommState failed with error %u.\n", GetLastError());
        return (2);
    }
    // Print state of the COM port after making changes to COM port.
    printCommState(&dcb);

    fprintf(stdout, "Serial port %s successfully reconfigured.\n", pcComPort);
    return(0); // Success status code.
}
