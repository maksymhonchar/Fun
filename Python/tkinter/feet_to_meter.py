import tkinter
from tkinter.ttk import Frame, Entry, Label, Button
from tkinter import N, W, E, S, StringVar


def calculate(parent):
    try:
        value = float(parent.feet_entry.get())
        parent.meters_value.set((0.3048 * value * 10000.0 + 0.5) / 10000.0)
    except ValueError:
        pass


class Program(Frame):
    """Class to represent a main window"""
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.meters_value = StringVar()
        self.initUI()

    def initUI(self):
        self.grid(column=0, row=0, sticky=(N, W, E, S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.parent.title('Feet to meter')
        self.pack(expand=True)
        self.centerWindow()
        self.initItems()

    def initItems(self):
        """Initialize all widgets here"""
        # An entry to enter feet value.
        self.feet_entry = Entry(self, width=7)
        self.feet_entry.grid(column=2, row=1, sticky=(W, E))
        # A label to display translated meters.
        self.meters_lbl = Label(self, textvariable=self.meters_value)
        self.meters_value.set(0)
        self.meters_lbl.grid(column=2, row=2, sticky=(W, E))
        # Different labels for text only.
        Label(self, text='feet').grid(column=3, row=1, sticky=W)
        Label(self, text='is equivalent to').grid(column=1, row=2, sticky=E)
        Label(self, text='meters').grid(column=3, row=2, sticky=W)
        # Button to calculate things.
        calc_btn = Button(self, text='Calculate',
                          command=lambda: calculate(self))
        calc_btn.grid(column=3, row=3, sticky=W)

        # This widget is just for fun.
        # Also it shows how to get an event callback when Entry widget is modified
        def callback(str):
            print(str.get())
        entry_str = StringVar()
        entry_str.trace('w', lambda name, index, mode, str=entry_str: callback(str))
        self.entry_widg = Entry(self, width=10, textvariable=entry_str)
        self.entry_widg.grid(column=1, row=4, sticky=(W, E))

        # A really simple way to change label text:
        test_lbl = Label(self)
        test_lbl.grid(column=3, row=4, sticky=E)
        test_lbl['text'] = 'hello!'

        # Handling label's events
        ev_lbl = Label(self, text='Do something with me...', width=30)
        ev_lbl.grid(column=1, row=5, columnspan=2, rowspan=2)
        ev_lbl.bind('<Enter>', lambda e: ev_lbl.configure(text='Moved mouse inside'))
        ev_lbl.bind('<Leave>', lambda e: ev_lbl.configure(text='Moved mouse outside'))
        ev_lbl.bind('<1>', lambda e: ev_lbl.configure(text='Clicked left mouse button!'))
        ev_lbl.bind('<Double-1>', lambda e: ev_lbl.configure(text='Double clicked left mouse button!'))

        # Configure pads for all grid cells
        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)
        # As default, entry is focused
        self.feet_entry.focus()

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
