import tkinter
from tkinter.ttk import Button, Style, Frame
from tkinter import RIGHT, BOTH, RAISED


class Program(Frame):
    """Class to represent a main window"""
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.style = Style()
        self.initUI()

    def initUI(self):
        self.parent.title('Restaurant manager')
        self.pack(fill=tkinter.BOTH, expand=1)
        self.style.configure('TFrame', background='#FF0000')
        self.initItems()
        self.centerWindow()

    def initItems(self):
        self.initItems_group2()
        self.initItems_group1()

    def initItems_group1(self):
        """Placing images with place() method"""
        # A simple button to quit the program.
        quitButton = Button(self, text='Quit', command=self.quit)
        quitButton.place(x=5, y=5)
        # A simple picture.
        google_pic = tkinter.PhotoImage(file='images/google.png')
        ggl_lbl = tkinter.Label(self, image=google_pic)
        ggl_lbl.image = google_pic
        ggl_lbl.place(x=95, y=5)
        # Example of zooming picture.
        yandex_pic = tkinter.PhotoImage(file='images/yandex.png')
        yandex_pic = yandex_pic.zoom(2)
        yandexpic_panel = tkinter.Label(image=yandex_pic)
        yandexpic_panel.image = yandex_pic
        yandexpic_panel.place(x=170, y=5)
        # Example of resizing picture.
        ya_pic = tkinter.PhotoImage(file='images/yandex.png')
        ya_pic = ya_pic.subsample(x=2, y=2)  # {x=1, y=1} is a original picture.
        yapic_panel = tkinter.Label(image=ya_pic)
        yapic_panel.image = ya_pic
        yapic_panel.place(x=240, y=5)
        # Add a separator for this and the next group.
        sep_pic = tkinter.PhotoImage(file='images/sep.png')
        sep_pic = sep_pic.subsample(x=1, y=1000)
        sep_lbl = tkinter.Label(image=sep_pic)
        sep_lbl.image = sep_pic
        sep_lbl.place(x=5, y=85)

    def initItems_group2(self):
        """Placing images with pack() method"""
        frame = Frame(self, relief=RAISED, borderwidth=1)
        frame.pack(fill=BOTH, expand=True)
        self.pack(fill=BOTH, expand=True)

        closeButton_2 = Button(self, text='Close!', command=self.quit)
        closeButton_2.pack(side=RIGHT, padx=5, pady=5)
        okButton = Button(self, text='OK')
        okButton.pack(side=RIGHT)

    def centerWindow(self):
        window_width = 300
        window_height = 150
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
