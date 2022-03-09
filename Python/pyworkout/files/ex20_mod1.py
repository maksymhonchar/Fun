from collections import defaultdict


def count_words() -> dict:
    stats = defaultdict(int)

    user_filepath = input('Enter name of text file: ')
    user_words_str = input('Enter words to count: ')

    try:
        user_words_lst = user_words_str.split()
        with open(user_filepath) as fs_r:
            for line in fs_r:
                for user_word in user_words_lst:
                    stats[user_word] += line.count(user_word)
    except FileNotFoundError:
        error_msg = f'count_words failed: cant open or find [{user_filepath}]'
        raise FileNotFoundError(error_msg)

    return stats


def main():
    stats = count_words()
    print(stats)


if __name__ == '__main__':
    main()
