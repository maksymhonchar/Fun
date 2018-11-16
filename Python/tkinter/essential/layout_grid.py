import tkinter
from tkinter.ttk import Frame, Button, Label
from tkinter import BOTH, Text, W, E, S, N


class Program(Frame):
    """Class to represent a main window"""
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.parent.title('Restaurant manager')
        self.pack(fill=BOTH, expand=True)
        self.initItems()
        self.centerWindow()

    def initItems(self):
        """Initialize all widgets and else here"""
        # Define some space among widgets in the grid.
        # [weight] param - makes the 2nd column and 4th row growable
        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)
        # Label sticks to the west.
        lbl = Label(self, text='Windows')
        lbl.grid(sticky=W, pady=4, padx=5)

        area = Text(self)
        # When window is resized, the text widget grows in all directions.
        area.grid(row=1, column=0, columnspan=2, rowspan=4,
                  padx=5, pady=5, sticky=E+W+S+N)

        # Create and add some buttons to the window.
        abtn = Button(self, text="Activate")
        abtn.grid(row=1, column=3)

        cbtn = Button(self, text="Close")
        cbtn.grid(row=2, column=3, pady=4)

        hbtn = Button(self, text="Help")
        hbtn.grid(row=5, column=0, padx=5)

        obtn = Button(self, text="OK")
        obtn.grid(row=5, column=3)

    def centerWindow(self):
        window_width = 500
        window_height = 500
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - window_width) / 2
        y = (screen_h - window_height) / 2
        self.parent.geometry('%dx%d+%d+%d' % (window_width, window_height, x, y))


def main():
    root = tkinter.Tk()
    app = Program(root)
    root.mainloop()


if __name__ == '__main__':
    main()
