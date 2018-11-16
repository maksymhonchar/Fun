import tkinter
from tkinter.ttk import Style, Frame, Label, Entry
from tkinter import TOP, BOTH, X, N, LEFT, RAISED, CENTER, RIGHT


class Program(Frame):
    """Class to represent a main window"""
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.style = Style()
        self.initUI()

    def initUI(self):
        self.parent.title('Restaurant manager')
        self.pack(fill=BOTH, expand=True)
        self.style.configure('TFrame', background='white')
        self.initItems()
        self.centerWindow()

    def initItems(self):
        """Create kinda complex layout, using a pack manager"""
        # Create and show the first frame.
        frame1 = Frame(self, relief=RAISED, borderwidth=1)
        frame1.pack(fill=X)
        # Add a label to first frame [frame1]
        lbl1 = Label(frame1, text='frame1 child', width=6)
        lbl1.pack(fill=X, padx=5, expand=True)  # (fill=x) overrides previous (justify=CENTER)
        # Add an entry to first frame [frame1]
        entry1 = Entry(frame1)
        entry1.pack(side=LEFT, padx=5, pady=5)  # default [side=CENTER]
        # Create and show the second frame
        frame2 = Frame(self, relief=RAISED, borderwidth=1)
        frame2.pack(fill=X)
        # Add a label to second frame [frame2]
        lbl2 = Label(frame2, text='frame2 child', width=14)
        lbl2.pack(side=RIGHT, padx=5, pady=5)
        # Add an entry to second frame [frame2]
        entry2 = Entry(frame2)
        entry2.pack(side=RIGHT, padx=5, pady=5)
        # Create and show the third frame
        frame3 = Frame(self, relief=RAISED, borderwidth=1)
        frame3.pack(fill=BOTH, expand=True)
        # Add a label and text widgets to the third frame
        lbl3 = Label(frame3, text='frame3 child', width=6)
        lbl3.pack(anchor=N, padx=5, pady=5)
        txt = tkinter.Text(frame3)
        txt.pack(fill=BOTH, pady=5, padx=5, expand=True)

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
