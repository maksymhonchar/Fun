import operator


def alphabetize_names(
    names: dict
) -> dict:
    return sorted(
        names,
        key=lambda item: [item['last'], item['first']]
    )


def alphabetize_names_v2(
    names: dict
) -> dict:
    return sorted(
        sorted(names, key=operator.itemgetter('first')),
        key=operator.itemgetter('last')
    )


def alphabetize_names_v3(
    names: dict
) -> dict:
    return sorted(
        names,
        key=operator.itemgetter('last', 'first')
    )


def main():
    phone_book = [
        {'first': 'Volodymir', 'last': 'Zelensky'},
        {'first': 'Volodymir', 'last': 'Veliky'},
        {'first': 'Volodymir', 'last': 'Abramov'},
        {'first': 'Ivan', 'last': 'Zelensky'},
        {'first': 'Anand', 'last': 'Zelensky'},
    ]

    result = alphabetize_names(phone_book)
    result_v2 = alphabetize_names_v2(phone_book)
    result_v3 = alphabetize_names_v3(phone_book)
    print(
        '[phone book]:\n', phone_book, '\n\n',
        '[result #1]:\n', result, '\n\n',
        '[result #2]:\n', result_v2, '\n\n',
        '[result #3]:\n', result_v3,
        sep=''
    )


if __name__ == '__main__':
    main()
