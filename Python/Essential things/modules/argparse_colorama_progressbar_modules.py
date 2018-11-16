#!/home/maxdev/Documents/pycharm_workspace/venv/bin/python

# An easy example of ArgumentParser from argparse example:
# python program.py >> Hello, World!
# python program.py Maxim >> Hello, Maxim!
def f1():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('name', nargs='?')
    args = ap.parse_args()
    name = (args.name or 'World')
    print('Hello', name, '!')


# Changing colors of text/background, changing styles
# with [colorama] module.
def f2():
    from colorama import Fore, Back, Style
    print(Fore.RED + 'some red text')
    print(Back.GREEN + 'green background here!')
    print(Style.BRIGHT + 'bright text here!')
    # Reset previous settings
    print(Fore.RESET + Back.RESET + Style.RESET_ALL)
    print('Usual text')


# Adding a progress bar
def f3():
    from progressbar import ProgressBar
    import time
    progress = ProgressBar()
    for i in progress(range(100)):
        time.sleep(0.01)
