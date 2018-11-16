#define LAMPS_MAX_COUNT 2

/*
   Defines for shiftOut method
*/
#define ON 1
#define OFF 0

/*
   Fields for shiftRegister #1.
*/
const int datapin = 2;
const int clockpin = 3;
const int latchpin = 4;

/*
   Class to describe lamp structure.
*/
class Lamp
{
  public:
    Lamp(int pinR_in, int pinG_in, int pinB_in)
    {
      this->pinR = pinR_in;
      this->pinG = pinG_in;
      this->pinB = pinB_in;
    }
    int pinR;
    int pinG;
    int pinB;
    int state = 0;
    int curColor = 0;
};

/*
   Global array of lamps to control all the lamps.
*/
Lamp *lampSet[LAMPS_MAX_COUNT] = {
  new Lamp(1, 2, 0),
  new Lamp(3, 4, 5),
  //new Lamp(6, 7, 8);
};

/*
   Byte to write into shift register.
   This byte also holds information what lamps are turned on/off.
*/
byte data = 0;

/*
   In initialization method, set shift register pins to OUTPUT
   and open serial default 9600 port.
*/
void setup()
{
  initShiftRegister();
  Serial.begin(9600);
}

/*
   In loop, rewrite lampSet array of Lamps and in the end
   update all settings.
*/
void loop()
{
  char incomingByte = Serial.read();
  switch (incomingByte)
  {
    case '1':
    case '2':
      switchLamp(lampSet[(incomingByte - '0') - 1]);
      break;
    case 'q':
    case 'w':
      changeColor(incomingByte, lampSet);
      break;
  }
  updateLamps(lampSet);
  printBits(data);
}

/*
   Switch lamp on, if it was turned off
   and vice versa.
*/
void switchLamp(Lamp *lamp)
{
  lamp->state = (lamp->state == ON) ? OFF : ON;
}

void changeColor(char incChar, Lamp *lampSet[])
{
  int index = (incChar == 'q') ? 0 : 1;
  switch (lampSet[index]->curColor)
  {
    case 0:
      lampSet[index]->curColor = 1;
      break;
    case 1:
      lampSet[index]->curColor = 2;
      break;
    case 2:
      lampSet[index]->curColor = 0;
      break;
  }
}

void updateLamps(Lamp *lampSet[])
{
  for (int i = 0; i < LAMPS_MAX_COUNT; i++)
  {
    if (lampSet[i]->state == ON)
    {
      switch (lampSet[i]->curColor)
      {
        case 0:
          bitWrite(data, lampSet[i]->pinR, ON);
          bitWrite(data, lampSet[i]->pinG, OFF);
          bitWrite(data, lampSet[i]->pinB, OFF);
          shiftOut(datapin, clockpin, MSBFIRST, data);
          digitalWrite(latchpin, HIGH);
          digitalWrite(latchpin, LOW);
          break;
        case 1:
          bitWrite(data, lampSet[i]->pinR, OFF);
          bitWrite(data, lampSet[i]->pinG, ON);
          bitWrite(data, lampSet[i]->pinB, OFF);
          shiftOut(datapin, clockpin, MSBFIRST, data);
          digitalWrite(latchpin, HIGH);
          digitalWrite(latchpin, LOW);
          break;
        case 2:
          bitWrite(data, lampSet[i]->pinR, OFF);
          bitWrite(data, lampSet[i]->pinG, OFF);
          bitWrite(data, lampSet[i]->pinB, ON);
          shiftOut(datapin, clockpin, MSBFIRST, data);
          digitalWrite(latchpin, HIGH);
          digitalWrite(latchpin, LOW);
          break;
      }
    }
    else
    {
      bitWrite(data, lampSet[i]->pinR, OFF);
      bitWrite(data, lampSet[i]->pinG, OFF);
      bitWrite(data, lampSet[i]->pinB, OFF);
      shiftOut(datapin, clockpin, MSBFIRST, data);
      digitalWrite(latchpin, HIGH);
      digitalWrite(latchpin, LOW);
    }
  }
}

void initShiftRegister()
{
  pinMode(datapin, OUTPUT);
  pinMode(clockpin, OUTPUT);
  pinMode(latchpin, OUTPUT);
}

void printBits(byte myByte) {
  for (byte mask = 0x80; mask; mask >>= 1) {
    if (mask  & myByte)
      Serial.print('1');
    else
      Serial.print('0');
  }
  Serial.println();
}

