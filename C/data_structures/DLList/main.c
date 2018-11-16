#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "list.h"

int main()
{
    list_t *testList = list_new();

    void *testValue = (void *)"TV1";
    void *testValue2 = (void *)"TV2";
    void *testValue3 = (void *)"TV3";
    void *testValue4 = (void *)"TV4";
    void *testValue5 = (void *)"TV5";

    list_addByIndex(testList, testValue, 0);
    list_addByIndex(testList, testValue2, 1);
    list_addByIndex(testList, testValue3, 2);
    list_addByIndex(testList, testValue4, 3);
    list_addByIndex(testList, testValue5, 4);

    // Print added items.
    for (int i = 0; i < list_getSize(testList); i++)
    {
        puts((char *)list_getNodeValueByIndex(testList, i));
    }

    puts("");
    // Changing SHIT MOTHERFUCKLERERRRRR
    void *changed = (void *)"FUCK YOU";
    list_setNodeByIndex(testList, 3, changed);
    for (int i = 0; i < list_getSize(testList); i++)
    {
        puts((char *)list_getNodeValueByIndex(testList, i));
    }

    list_delete(testList);
    return 0;
}
