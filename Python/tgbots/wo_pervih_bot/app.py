# -*- coding: utf-8 -*-
import logging
from telegram.ext import Updater, CommandHandler, Filters, MessageHandler


class TGBot(object):
    """
    maksymhonchar bot instance. init, logging, subscribing and polling.
    """
    # todo: timeout
    # todo: deployment

    def __init__(self, token, main_handlers):
        logging.basicConfig(format='%(name)s - %(message)s', level=logging.DEBUG)
        self.updater = Updater(token=token)
        self.handlers = main_handlers
        self._init_handlers()

    def run(self):
        self.updater.start_polling()

    def _init_handlers(self):
        text_handler = MessageHandler(Filters.text, self.handlers.handle_text)
        sticker_handler = MessageHandler(Filters.sticker, self.handlers.handle_sticker)
        start_handler = CommandHandler('start', self.handlers.handle_start)
        self.updater.dispatcher.add_handler(start_handler)
        self.updater.dispatcher.add_handler(sticker_handler)
        self.updater.dispatcher.add_handler(text_handler)


class Handler(object):
    """
    Main handlers for maksymhonchar bot.
    """

    def __init__(self):
        self.start_h = self.handle_start
        self.sticker_h = self.handle_sticker
        self._text_h = []

    def append(self, item):
        handler_not_duplicate = item not in self._text_h
        if handler_not_duplicate:
            self._text_h.append(item)

    def handle_text(self, bot, update):
        """global text handler"""
        for handler in self._text_h:
            if handler(bot, update):
                break

    @staticmethod
    def handle_sticker(bot, update):
        """global sticker handler"""
        reply_to_msg = update.message.message_id
        bot.sendMessage(chat_id=update.message.chat_id, text='классный стикер!', reply_to_message_id=reply_to_msg)

    @staticmethod
    def handle_start(bot, update):
        bot.sendMessage(chat_id=update.message.chat_id, text='поїхали водій!')


class TextHandler(object):
    """
    Handlers for maksymhoncharbot bot commands.
    """
    phrases = [  # todo: phrases from file
        ['во-первых', 'во первых', 'по-перше', 'во-1', 'вопервых', 'поперше', 'по перше', 'по-1'],
        ['во-вторых', 'во вторых', 'по-друге', 'во-2', 'вовторых', 'по друге', 'по-2'],
        ['в третьих', 'втретьих', 'в-третьих', 'по-третє', 'в-3', 'по третє', 'по-3'],
        ['стар\nтап'],
        ['кто такой саня?'],
    ]
    answers = [  # todo: answers as a set + from file
        'Во 1) что же ты мне сделаешь',
        'во 2) вовторых ухади отсюда',
        'во3) что ты мне сделаешь, я в другмо городе за мат извени.',
        'с-т-а-р-т-а-п\nс\nт\nа\nр\nт\nа\nп\n',
        '?? по-моему, он просто красавчик. ' + u'\U0001F4A8 ' +' him ' + u'\U0001F525' + u'\U0001F525',
    ]

    @staticmethod
    def handle_dummy(bot, update):
        # todo: method for concrete question-answer/set of
        text = update.message.text.lower().strip()
        if 'плохой бот' in text or 'тупой бот' in text:
            from_obj = update.message.from_user
            to_send_2 = '{} {} - ну ты чего'.format(from_obj.first_name, from_obj.last_name)
            bot.sendMessage(chat_id=update.message.chat_id, text=to_send_2)
            return True
        if 'дай стикер!' in text:
        	bot.sendSticker(chat_id=update.message.chat_id, sticker='AAQCABPbTksNAARJdZYf3w0dpiWWAQABAg')
        	return True
        return False

    def handle_wo_pervih(self, bot, update):
        text = update.message.text.lower().strip()
        for index, phrases_list in enumerate(self.phrases):
            phrase_in_text = [p in text for p in phrases_list]
            if any(phrase_in_text):
                msg_to_reply = update.message.message_id
                text_to_send = self.answers[index]
                bot.sendMessage(chat_id=update.message.chat_id, text=text_to_send, reply_to_message_id=msg_to_reply)
                return True
        return False


def main():
    # todo: different structure (!)

    txt_handlers = TextHandler()
    main_handlers = Handler()

    main_handlers.append(txt_handlers.handle_dummy)
    main_handlers.append(txt_handlers.handle_wo_pervih)

    bot = TGBot('token', main_handlers)

    bot.run()


if __name__ == '__main__':
    main()
