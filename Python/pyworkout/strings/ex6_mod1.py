import random
import string


def create_content() -> str:
    lines = []
    lines_cnt = 10
    for _ in range(lines_cnt):
        words = []
        words_cnt = random.randint(5, 10)
        for _ in range(words_cnt):
            chars_cnt = random.randint(5, 10)
            word = ''.join( random.choices(string.ascii_letters, k=chars_cnt) )
            words.append(word)
        
        lines_separator = '\n'
        words.append(lines_separator)

        words_separator = ' '
        line = words_separator.join(words)
        lines.append(line)

    random_text = ''.join(lines)

    return random_text


def create_file(
    content: str
) -> str:
    random_filename = ''.join( random.choices(string.ascii_letters, k=30) ) + '.txt'

    with open(random_filename, 'w') as fs_w:
        fs_w.write(content)
        
    return random_filename


def read_and_display(
    filepath: str,
    n: int
) -> None:
    print(f'Displaying every {n}th+ words:')

    with open(filepath, 'r') as fs_r:
        for line in fs_r:
            words_separator = ' '
            words = line.split(sep=words_separator)
            nth_word_and_further = words_separator.join( words[n-1:] )
            
            print(f'[{line.strip()}] --> [{nth_word_and_further.strip()}]')


def main() -> None:
    content = create_content()
    random_filename = create_file(content)
    read_and_display(random_filename, n=3)


if __name__ == '__main__':
    main()
