import os
import string


def separate_chars(
    input_file_path: str,
    vowels_file_path: str,
    consonants_file_path: str
) -> None:
    vowels = {'a', 'e', 'i', 'o', 'u'}
    consonants = set(string.ascii_lowercase) - vowels

    with open(input_file_path) as fs_r:
        with open(vowels_file_path, 'w', newline='') as vowels_fs_w, open(consonants_file_path, 'w', newline='') as cons_fs_w:
            for line in fs_r:
                line_lower = line.lower()

                line_vowels = ''.join(
                    [char for char in line_lower if char in vowels]
                ) + os.linesep
                vowels_fs_w.write(line_vowels)

                line_consonants = ''.join(
                    [char for char in line_lower if char in consonants]
                ) + os.linesep
                cons_fs_w.write(line_consonants)


def main():
    input_file_path = 'ex24_mod2.py'
    vowels_file_path = 'new_file_1.txt'
    consonants_file_path = 'new_file_2.txt'
    separate_chars(input_file_path, vowels_file_path, consonants_file_path)


if __name__ == '__main__':
    main()
