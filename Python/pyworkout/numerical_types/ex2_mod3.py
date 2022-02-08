def get_words_stats(
    words: list
) -> tuple:
    words_len = len(words)
    words_is_empty = words_len == 0
    if words_is_empty:
        return None, None, None

    shortest_word_len = 0
    longest_word_len = 0
    average_word_len = 0

    for word in words:
        word_len = len(word)
        if word_len < shortest_word_len:
            shortest_word_len = word_len
        if word_len > longest_word_len:
            longest_word_len = word_len
        average_word_len += word_len / words_len

    return shortest_word_len, longest_word_len, average_word_len


if __name__ == '__main__':
    print( get_words_stats(['', 'a', 'bb', 'ccc', 'dddd', 'eeeee']) )
