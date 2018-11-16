from tkinter import BOTH, Tk
from tkinter.ttk import Frame, Style, Label
from tkinter import PhotoImage
# Python Imaging Library (PIL)


class MainApplication(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.style = Style()
        self.initUI()

    def initUI(self):
        self.parent.title('Absolute positioning')
        self.pack(fill=BOTH, expand=1)
        self.style.configure('TFrame', background='grey')

        apple_png = PhotoImage(file='images/nibbles/apple.png')
        label1 = Label(self, image=apple_png)
        label1.image = apple_png
        label1.place(x=20, y=20)

        dot_png = PhotoImage(file='images/nibbles/dot.png')
        label2 = Label(self, image=dot_png)
        label2.image = dot_png
        label2.place(x=40, y=160)

        head_png = PhotoImage(file='images/nibbles/head.png')
        label3 = Label(self, image=head_png)
        label3.image = head_png
        label3.place(x=170, y=50)


def main():
    root = Tk()
    root.geometry('300x280+300+300')
    app = MainApplication(root)
    root.mainloop()


if __name__ == '__main__':
    main()