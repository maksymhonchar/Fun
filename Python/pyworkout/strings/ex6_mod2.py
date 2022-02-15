def transpose_strings(
    strings: list
) -> list:
    words_separator = ' '

    transposed_words = []
    for _ in range(len(strings)):
        transposed_words.append([])

    for string in strings:
        words = string.split(sep=words_separator)
        for col_idx, word in enumerate(words):
            transposed_words[col_idx].append(word)

    transposed_strings = []
    for words in transposed_words:
        transposed_string = words_separator.join(words)
        transposed_strings.append(transposed_string)

    return transposed_strings


def main() -> None:
    before = ['abc def ghi', 'jkl mno pqr', 'stu vwx yz']
    after = transpose_strings(before)
    print(f'Transposing results:\n{before} --> {after}')


if __name__ == '__main__':
    main()
