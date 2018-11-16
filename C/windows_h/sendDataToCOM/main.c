/*
DWORD: 32-bit uint (typedef unsigned long DWORD)
HANDLE: a handle to an object (typedef PVOID HANDLE)
PVOID: pointer to any type (typedef void *PVOID)
DCB [structure]: defines the control setting for a serial communications device.
COMMTIMEOUTS [structure]: Contains the time-out parameters for a communications device. The parameters determine the behavior of ReadFile, WriteFile, ReadFileEx, and WriteFileEx operations on the device.

*/
#include <stdio.h>
#include "Utils.h"

int main()
{
    // Define the bits to send.
    char bytes_to_send[6] = {'h', 'e', 'l', 'l', 'o'};

    // Declare variables and structures.
    HANDLE hSerial;
    DCB dcbSerialParams = {0};
    COMMTIMEOUTS timeouts = {0};

    // Open the port number.
    openPort(&hSerial);

    // Set device parameters.
    // {38400 baud; 1 start bit; 1 stop bit; no parity}
    setDeviceParameters(&dcbSerialParams, &hSerial);

    // Set COM port timeout settings.
    setTimeouts(&timeouts, &hSerial);

    // Send specified text.
    DWORD bytes_written = 0, total_bytes_written = 0;
    fprintf(stderr, "Sending bytes...");
    if(!WriteFile(hSerial, bytes_to_send, 5, bytes_to_send, NULL))
    {
        fprintf(stderr, "Error by sending bytes.\n");
        CloseHandle(hSerial);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "%d bytes written.\n", bytes_written);

    // Close a serial port.
    closePort(&hSerial);

    return (EXIT_SUCCESS);
}
