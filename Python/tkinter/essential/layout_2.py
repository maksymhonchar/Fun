import tkinter
from tkinter.ttk import Frame, Label, Separator
from tkinter import X, StringVar


class Program(Frame):
    """Class to represent a main window"""
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.pack(fill='both', expand=True)
        self.parent.title('Test application')
        self.centerWindow()
        self.initItems()

    def initItems(self):
        """Initialize all widgets here"""
        # First way to create a frame
        frame = Frame(self)
        frame['padding'] = (5, 10)
        frame['borderwidth'] = 2
        frame['relief'] = 'sunken'
        frame.pack(fill='both')
        lbl = Label(frame, text='sunken!')
        lbl.pack()
        lbl2 = Label(frame, text='_2!')
        lbl2.pack()

        # The second way to create a frame
        frame2 = Frame(self, borderwidth=2, relief='raised')  # or constant tkinter.RAISED
        frame2.pack(fill='both')
        Label(frame2, text='raised').grid(row=1, column=1, sticky='w')
        Label(frame2, text='2col 2row').grid(row=2, column=2, sticky='we')
        Label(frame2, text='3col 3row', relief='raised', anchor='center',
              wraplength=50, width='7').grid(row=3, column=3, columnspan=2, sticky='we')
        Label(frame2, text='5col 5row').grid(row=4, column=5, sticky='e')

        # The first way to create a separator - with Frame
        sep1 = Frame(self, height=2, borderwidth=1, relief='sunken')
        sep1.pack(fill=X, padx=5, pady=10)
        Label(self, text='Between the separators').pack()
        # The second way to create a Separator - with tkinter.ttk.Separator
        sep = Separator(self, orient=tkinter.HORIZONTAL)
        sep.pack(fill=X, padx=5, pady=10)

    def centerWindow(self):
        """Place the main window in the center of screen"""
        window_w = 300
        window_h = 200
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - window_w) / 2
        y = (screen_h - window_h) / 2
        self.parent.geometry('%dx%d+%d+%d' % (window_w, window_h, x, y))


def main():
    root = tkinter.Tk()
    app = Program(root)
    root.mainloop()


if __name__ == '__main__':
    main()
