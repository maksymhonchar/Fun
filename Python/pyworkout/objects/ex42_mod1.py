from typing import Any


class StringKeyDict(dict):

    def __setitem__(
        self,
        key: Any,
        value: Any
    ) -> None:
        return super().__setitem__(str(key), value)


def main():
    d = StringKeyDict()
    d[1] = 1
    print(d)


if __name__ == '__main__':
    main()
