"""
copypaste from habr tutorial
https://habrahabr.ru/post/316666/


- dialog
- using markdown
- using inline keyboard
- using inline key answers

- yield technique for bot dialog

"""

from bot import DialogBot
from dialog import dialog


def main():
    token = '123'

    print('start ini')
    dialog_bot = DialogBot(token, dialog)
    print('start bot')
    dialog_bot.start()


if __name__ == '__main__':
    main()
