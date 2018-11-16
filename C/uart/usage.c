typedef unsigned char u8;

/* Check amount of bytes in the buffer */
u8 idxDiff(u8 idxIN, u8 idxOUT, u8 bufsize)
{
  if (idxIN >= idxOUT)
    return (idxIN - idxOUT);
  else
    return ((bufsize - idxOUT) + idxIN);
}

/* Track receiving and timeouts */
/* Drops buffer or switches its mode. */
/* Calls in the main infinite loop */
void usartPool(void)
{
  u8 rxcnt = idxDiff(inrx, outrx, RXBUF_SIZE);
  if (0 == rxcnt)
  {
    /* There is nothing in buffer */
    if (rxtimeout >= BLOCK_TIMEOUT)
    {
      /* There is a while when it was something in buffer */
      ...
    }
  }
  else if (rxcnt < BLOCK_SIZE)
  {
    /* There isn't much in buffer */
    if (rxtimeout >= BYTE_TIMEOUT)
    {
        /* Too big pause between bytes */
        /* Drop the buffer */
        outrx = inrx;
    }
  }
  else
  {
    /* If BLOCK_SIZE byte was received correctly */
    /* Work on the data */
    ...
  }
}
