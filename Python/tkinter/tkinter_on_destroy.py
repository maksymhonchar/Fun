from tkinter import *
import _tkinter

root = Tk()

print('Tkinter pyd path is:', _tkinter.__file__)

def _delete_window():
    print('delete window.')
    try:
        root.destroy()
    except:
        print('Cannot destroy root. Pass')
        pass

def _destroy(event):
    print(event, 'call from _destroy() function')

root.protocol('WM_DELETE_WINDOW', _delete_window)
root.bind('<Destroy>', _destroy)

button =  Button(root, text='Destroy', command=root.destroy)
button.pack()

mainloop()
