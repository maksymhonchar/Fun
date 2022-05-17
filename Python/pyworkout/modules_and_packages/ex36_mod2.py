from typing import Dict


def apply_methods(
    data: str
) -> Dict[str, int]:
    methods = {
        'isdigit': str.isdigit,
        'isalpha': str.isalpha,
        'isspace': str.isspace
    }
    return {
        method_name: sum(
            [
                method_func(char)
                for char in data
            ]
        )
        for method_name, method_func in methods.items()
    }


def main():
    for data in ('hello', '123', ' '):
        result = apply_methods(data)
        print(f'[{data=}] [{result=}]')


if __name__ == '__main__':
    main()
