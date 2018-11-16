#include <avr/io.h>
#include <stdio.h>

#define True (1)

typedef unsigned char u8;
typedef unsigned short int u16;

void uartInit(void);
u8 rxUART(void);
void txUART(u8);

int main()
{
  u8 c;
  u16 i;

  uartInit();
  while(True)
  {
    c = rxUART();
    putchar(c);
    txUART(c);
  }
}

void uartInit(void)
{
  /* Set baud rate */
  u16 baud = 11;
  UBBRH = (unsigned char)(baud>>8);
  UBBRL = (unsigned char)baud;
  /* Enable receiver and transmitter */
  UCSRB = (1<<RXEN) | (1<<TXEN);
  /* Set frame format: 8data, 2stop bit */
  UCSRC = (1<<URSEL) | (1<< USBS) | (3<<UCSZO);
}

u8 rxUART(void)
{
  /* Wait for data to be received */
  while(!(UCSRA & (1 << RXC)));
  /* Get and return received data from buffer */
  return UDR;
}

void txUART(void)
{
  /* Wait for empty transmit buffer */
  while(!(UCSRA & (1 << UDRE)));
  /* Put data into buffer, send the data */
  UDR = d;
}


/* Utility to fill a hex digit (as uchar) to the char array */
/* Note, that you should free memory after using returned char array */
u8 getCharArrFromHex(u8 hexDigit)
{
  u8 *digitsArr = calloc(4, sizeof(u8));
  sprintf(digitsArr, "%x", hexDigit);

  u8 decNum = 0;
  sscanf(digitsArr, "%d", decNum);

  return decNum;
}


/* Utility routine to read two hex digits and convert them into a u8 */
u8 getU8(void)
{
  u8 x = 0;
  u8 hexDigit1 = rxUART();
  u8 hexDigit2 = rxUART();

  u8 decDigit1 = getCharArrFromHex(hexDigit1);
  u8 decDigit2 = getCharArrFromHex(hexDigit2);

  x |= (decDigit1 << 4);
  x |= decDigit2;

  return x;
}
