#include <stdlib.h>
#include <stdio.h>

#include "stack.h"

// private:
struct stack_s
{
    char *data[DATA_MAXSIZE];
    int size;
};

stack_t *stack_new(void)
{
    stack_t *out = (stack_t *)malloc(sizeof(struct stack_s));
    out->size = 0;
    for(int i = 0; i < DATA_MAXSIZE; i++)
    {
        out->data[i] = NULL;
    }
    return (out);
}

void stack_delete(stack_t * self)
{
    free(self);
}

void stack_push(stack_t * self, char *element)
{
    if(self->size == DATA_MAXSIZE)
        return;
    self->data[self->size] = element;
    self->size++;
}

char *stack_pop(stack_t * self)
{
    char *toRet;
    if(self->size == 0)
    {
        toRet = "No items in stack.";
        return (toRet);
    }
    toRet = self->data[self->size - 1];
    self->data[self->size - 1] = NULL;
    self->size--;
    return (toRet);
}

char *stack_top(const stack_t * self)
{
    return (self->data[self->size]);
}

void stack_print(const stack_t * self)
{
    if(self->size == 0)
    {
        fprintf(stderr, "No items in stack.\n");
        return;
    }
    for(int i = 0; i < self->size; i++)
    {
        printf("%2i: %s\n", (i + 1), self->data[i]);
    }
}

int stack_getSize(const stack_t * self)
{
    return (self->size);
}
