import tkinter
from tkinter.ttk import Frame, Label, Checkbutton, Button, Entry, Style
from tkinter import BooleanVar, N, S, W, E


class Program(Frame):
    """Class to represent a main window"""
    def __init__(self, parent):
        Style().configure('A.TFrame', background='white')
        Frame.__init__(self, parent)
        self.grid(sticky=N+S+E+W)
        self.parent = parent
        self.initWidgets()

    def initWidgets(self):
        """Initialize all widgets here"""
        # Frame to contain all items in app.
        content = Frame(self, padding=(3, 3, 12, 12))
        content.grid(column=0, row=0, sticky=N+S+E+W)

        # Init default variables values
        onevar = BooleanVar()
        twovar = BooleanVar()
        threevar = BooleanVar()
        onevar.set(True)
        twovar.set(True)
        threevar.set(True)

        # Widgets initializing here:
        frame = Frame(content, borderwidth=5, relief='sunken', width=200, height=100)
        name_lbl = Label(content, text='Name')
        name_entry = Entry(content)
        one_chbtn = Checkbutton(content, text='One', variable=onevar, onvalue=True)
        two_chbtn = Checkbutton(content, text='Two', variable=twovar, onvalue=True)
        three_chbtn = Checkbutton(content, text='Three', variable=threevar, onvalue=True)
        ok_btn = Button(content, text='Okay', command=lambda: print('hello!'))
        cancel_btn = Button(content, text='Cancel', command=self.quit)

        # Place all widgets on the grid.
        frame.grid(column=0, row=0, columnspan=3, rowspan=2, sticky=N+S+E+W)
        name_lbl.grid(column=3, row=0, columnspan=2, sticky=N+W, padx=5)
        name_entry.grid(column=3, row=1, columnspan=2, sticky=N+W, pady=5, padx=5)
        one_chbtn.grid(column=0, row=3)
        two_chbtn.grid(column=1, row=3)
        three_chbtn.grid(column=2, row=3)
        ok_btn.grid(column=3, row=3)
        cancel_btn.grid(column=4, row=3)

        # NEVER forget to configure first col and row weight!
        top = self.winfo_toplevel()
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Configure other columns and rows weights
        weight_list = [3, 3, 3, 1, 1]
        for i, weight in enumerate(weight_list):
            content.columnconfigure(i, weight=weight)
        content.rowconfigure(1, weight=1)

        content.rowconfigure('all', minsize=50)
        content.columnconfigure('all', minsize=50)


def main():
    root = tkinter.Tk()
    # To restrict resizing, use the following:
    root.resizable(False, True)  # Width is restricted, Height is still allowed.
    app = Program(root)
    root.mainloop()


if __name__ == '__main__':
    main()

