from typing import Any


class FlexibleDict(dict):

    def __getitem__(
        self,
        key: Any
    ) -> Any:
        if key in self:
            pass
        elif isinstance(key, str) and key.isnumeric() and (int(key) in self):
            key = int(key)
        elif isinstance(key, int) and (str(key) in self):
            key = str(key)
        else:
            error_msg = f"{key=} does not exist"
            raise KeyError(error_msg)

        return super().__getitem__(key)


def main():
    fdict = FlexibleDict({1: 100, 2: 200, 3: 300, 'hi': 'hello!'})
    print(f"{fdict}, {fdict[1]=}, {fdict['1']=}")


if __name__ == '__main__':
    main()
