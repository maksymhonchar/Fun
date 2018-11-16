typesofpeople = 10
x = "There are %d types of people" % typesofpeople
firstint = 5
secondint = 10
y = "There are two integers: %d and %d" % (firstint, secondint)

print(x)
print(y)

sstr = "Mark sayed: %r"
print(sstr % "hi jaine!")
print(sstr % False)

str1 = "contatination"
str2 = "is cool"
str3 = str1 + " " + str2
print(str3)

intrst = "It`s really interesting: %s"
lie = intrst % 'just a joke lul'
print(lie)

rawinput = "In the right is the raw input: %r"
testllist = [1,2,3,4]
rawinputwithlist = rawinput % testllist
print(rawinputwithlist)
rawinputwithstring = rawinput % 'hello'
print(rawinputwithstring)

# Note: Raw input doesn't affect regular integers like above
rawinputwithinteger = rawinput % 123
print(rawinputwithinteger)

lotofdots = '.' * 20;
print("20 dots, sir:", lotofdots)

# Comma at the end of print
a1 = "1234567890"
a2 = 'qwerty'
a3 = 'asdfgh'
a4 = 'zxcvbn'
print(a1 + a2, end=' ')
# Unfortunately, comma separator at the end of println
# statement doesn't work
print(a3 + a4)
print("hello there!")

print('double quotes "fuck you" - he told.\n\r "No, fuck you!"')

# for i in range(200):
#    print('\a', end=' ')

# iterator_justtonotusetrueloop = 0
# iterator_justtonotusetrueloop < 100
#while True:
#    for i in ["/","-","|","\\","|"]:
#        print("%s\r" % i, end=' ')
#        iterator_justtonotusetrueloop += 1

# Another one example

b = 189.5
sentence1 = "this is very long sentence"
print(round(b))
reversedSentence1 = ''.join(reversed(sentence1))
print(sentence1, "\n", reversedSentence1)
reversSente2 = ""
for character in reversed(sentence1):
    reversSente2 += character
print(reversSente2)

S = 'asdfghjk'
for i in range(0, len(S), 2):
    print(S[i], end=' ')
for i in S[::2]:
    print(i, end=' ')
