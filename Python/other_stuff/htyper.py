from random import randint
import sys, tty, termios


def getch():
    # getch function to read one character, without
    # pressing [enter] button.
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main():
    # Get content as a list of paths to files.
    filename = sys.argv[1:]
    if not filename:
        print('Usage: python htyper.py [file_to_hack]')
        return 1
    print('\nPress [ESC] to quit the program.\n')

    # Loop is needed, because if file has reached the end,
    # program will read it from the beginning.
    while True:
        # Open a file stream to read content.
        try:
            fs = open(filename[0])
        except Exception as e:
            print('Error opening file:', e.args[1])
            return
        # While there is a text in file stream, read
        # and print several amount characters.
        content = fs.read(randint(1, 5))
        while content:
            content = fs.read(randint(1, 4))
            # Wait for users next keystroke.
            ch = getch()
            if ord(ch) == 27:  # ESC
                return
            # Print content into command line.
            print(content, end='', flush=True)
        # When there is no text in file, close the file
        # and read it from the beginning.
        fs.close()


if __name__ == '__main__':
    main()
