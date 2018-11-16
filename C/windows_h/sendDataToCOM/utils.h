#pragma once

#include <windows.h>
#include <stdio.h>

void openPort(HANDLE *hSerial);
void setDeviceParameters(DCB *dcbSerialParams, HANDLE *hSerial);
void setTimeouts(COMMTIMEOUTS *timeouts, HANDLE *hSerial);
void closePort(HANDLE *hSerial);
