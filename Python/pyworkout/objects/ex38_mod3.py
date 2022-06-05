class LogFile(object):

    def __init__(
        self,
        filename: str
    ) -> None:
        self.filename = filename
        self.file = open(filename, 'w')


def main():
    filename = 'file.txt'
    logfile = LogFile(filename)

    result = logfile.file.closed
    print(result)


if __name__ == '__main__':
    main()
