#include <strings.h>
#include <stdio.h>
#include <windows.h>
#include <unistd.h>

#include "COMUtils.h"


int main(int argc, char **argv)
{
    // COM port handler.
    BOOL initFailed = initCOM();
    if (initFailed)
    {
        fprintf(stderr, "COM port initialization failed.\n");
        exit(EXIT_FAILURE);
    }

    // Variables to handle received data.
    char tmp;
    char buffer[256];
    ZeroMemory(buffer, sizeof(buffer));
    DWORD NoBytesRead;
    int seconds = 2;

    // Pipe for the gnuplot.
//    FILE *plotPipe = _popen("C:/gnuplot/bin/gnuplot.exe", "w");
//    fprintf(plotPipe, "load \"config.plt\"\n");
//    fflush(plotPipe);

    // Stream to save values to the plot.
    FILE *plotFS = fopen("plot.dat", "wa");
    fprintf(plotFS, "0.0,0.0\n1.0,0.0\n");

    // Start listening to the COM port.
    printf("Listening to the port...\n");
    do
    {
        // Wait for the incoming byte.
        ReadFile(hCom,
                 &tmp,
                 sizeof(tmp),
                 &NoBytesRead,
                 NULL );

        int cntr = 0;
        ZeroMemory(buffer, sizeof(buffer));
        while(tmp != 'x')
        {
            buffer[cntr++] = tmp;
            ReadFile(hCom,
                 &tmp,
                 sizeof(tmp),
                 &NoBytesRead,
                 NULL );
        }
        // Save received values.
        fprintf(plotFS, "%d,%s\n", seconds++, buffer);
        fflush(plotFS);

        // Print buffer + collected character.
        printf("Received a word: %s\n", buffer);
    }
    while (NoBytesRead > 0);

    // Close file stream for writing plot data.
    fclose(plotFS);
//    _pclose(plotPipe);

    printf("End of the program");
    return(EXIT_SUCCESS);
}
