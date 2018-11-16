#pragma once

#include <windows.h>

HANDLE hCom;
DCB dcb;

DWORD sendData(const char *data, DWORD size);
void printCommState(DCB *dcb);
int initCOM();
