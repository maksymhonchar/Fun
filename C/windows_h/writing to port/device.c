#include <strings.h>
#include <stdio.h>
#include <windows.h>
#include <math.h>
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

    // Prepare buffer for sin values.
    // 6 digits + 1 ending signal + 1 dot + 1 {\0} character + 1 {-] character.
    char valueToSend[10];
    ZeroMemory(valueToSend, sizeof(valueToSend));
    // X-Axis values.
    double xValue = 0;

    // Start the getting-sending loop.
    while(strcmp(valueToSend, "--exit"))
    {
        // Generate sin(sec) function.
        sprintf(valueToSend, "%.5fx", sin(xValue));
        xValue += 0.5;
        usleep(5000000);

        // Send the data.
        int size = sendData(valueToSend, strlen(valueToSend));
        if(size == strlen(valueToSend)
           ||
           ((size != strlen(valueToSend) + 1) && atof(valueToSend) < 0))
            printf("Word: %s. Bytes: %d.\n", valueToSend, size);
        else
            printf("Error handled by sending.\n");

    }

    printf("End of the program.");
    return(EXIT_SUCCESS);
}
