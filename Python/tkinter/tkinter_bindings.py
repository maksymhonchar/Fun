from tkinter import *

root = Tk()

def click_cb(event):
    # you have to click in the frame before it starts receiving any keyboard events.
    frame.focus_set()
    print('Clicked at', event.x, event.y)

def key_cb(event):
    # catches both latin and cyrillic characters.
    print('Pressed', repr(event.char))

def help_cb(event):
    print('call from help_cb')

frame = Frame(root, width=100, height=100)
frame.bind('<Key>', key_cb)  
frame.bind('<Button-1>', click_cb)
frame.pack()

root.bind_all('<Key-F1>', help_cb)

root.mainloop()
