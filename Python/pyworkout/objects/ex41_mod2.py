class Phone:

    def dial(
        self,
        number: str
    ) -> str:
        msg = f'{number} DIALING OK'
        return msg


class SmartPhone(Phone):

    def run_app(
        self,
        number: str
    ) -> None:
        dial_msg = self.dial(number)
        msg = f'[app] [status] dialing result: [{dial_msg}]'
        print(msg)


class iPhone(SmartPhone):

    def dial(
        self,
        number: str
    ) -> str:
        dial_msg = super().dial(number)
        msg = dial_msg.lower()
        return msg

    def run_app(
        self,
        number: str
    ) -> None:
        print('[app] [alert] iPhone message:')
        super().run_app(number)


def main():
    number = "+380991234567"

    sp = SmartPhone()
    sp.run_app(number)

    ip = iPhone()
    ip.run_app(number)


if __name__ == '__main__':
    main()
