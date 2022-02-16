def sort_sentence(
    sentence: str
) -> str:
    words_separator = ' '
    words = sentence.split(sep=words_separator)

    new_words_separator = ','
    sorted_sentence = new_words_separator.join( list(sorted(words)) )
    return sorted_sentence


def main() -> None:
    input = "Tom Dick Harry"
    print( sort_sentence(input) )


if __name__ == '__main__':
    main()
