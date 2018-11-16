#!/usr/bin/python3
# -*- coding: utf-8 -*-


from tkinter import Tk, BOTH
# Widgets that are themed can be imported from the ttk module.
from tkinter.ttk import Frame, Button, Style


class Example(Frame):
    if __name__ == '__main__':
        def __init__(self, parent):
            Frame.__init__(self, parent)

            # Save reference to the parent widget
            # Now it's a Tk root window.
            self.parent = parent

            # Create UI elements in this method.
            self.init_UI()

        def init_UI(self):
            # Set the title of the window.
            self.parent.title('Testing title')

            # TODO: comment here
            self.style = Style()
            self.style.theme_use('default')

            # Pack method = geometry manager.
            # Organizes widgets to horizontal and vertical boxes.
            # So now it takes the whole client space of the root window.
            self.pack(fill=BOTH, expand=1)

            # Center a window
            self.center_window()

            # Create a button
            quit_btn = Button(self, text='Quit', command=self.quit)
            quit_btn.place(x=50, y=50)

	    # Styling labels example
	    Style().configure('A.TLabel', foreground='black', background='white')
            Style().configure('B.TLabel', foreground='white', background='grey')
            # Is this approach compact enough?
            text_in_labels = ['first', 'second', 'third', 'fourth']
            for i in range(len(text_in_labels)):
            def_style = 'TLabel'
            if i % 2 == 0:
                def_style = 'A.TLabel'
            else:
                def_style = 'B.TLabel'
            w = Label(self, text=text_in_labels[i], style=def_style)
            w.pack(fill='x')

        def center_window(self):
            margin_left = 290
            margin_top = 150
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            x = (screen_w - margin_left) / 2
            y = (screen_h - margin_top) / 2
            self.parent.geometry('%dx%d+%d+%d' %
                                 (margin_left, margin_top, x, y))


def main():
    # Create a root window - a container for other widgets.
    # Root window is a main application window for program.
    # It must be created before other widgets.
    root = Tk()

    # Sets a size for the window and positions it on the screen.
    # [width]x[height] [x]screenCoordinate [y]screenCoordinate
    # root.geometry('250x150+300+300')

    # Create the instance of the application class
    app = Example(root)

    # Enter the mainloop.
    # Mainloop receives events from window system and
    # dispatches them to the application widgets.
    root.mainloop()


if __name__ == '__main__':
    main()

