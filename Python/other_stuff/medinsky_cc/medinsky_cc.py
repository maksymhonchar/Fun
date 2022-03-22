import os


def validate(code):
    """src: https://ru.wikipedia.org/wiki/Алгоритм_Луна"""
    LOOKUP = (0, 2, 4, 6, 8, 1, 3, 5, 7, 9)
    code = ''.join([char for char in code if char.isdigit()])
    evens = sum(int(i) for i in code[-1::-2])
    odds = sum(LOOKUP[int(i)] for i in code[-2::-2])
    return ((evens + odds) % 10 == 0)


def get_candidates(
    mask: str
) -> list:
    digits_to_guess = mask.count('*')
    candidates = []
    for guess in range(10**digits_to_guess):
        to_replace = '*' * digits_to_guess
        replace_with = str(guess).rjust(digits_to_guess, '0')
        candidate = mask.replace(to_replace, replace_with)
        if validate(candidate):
            candidates.append(candidate)
    return candidates


def write_cc_data(
    file_path: str,
    cc_numbers: list,
    exp_year: str,
    exp_month: str
) -> None:
    with open(file_path, 'w', newline='') as fs:
        for cc_number in cc_numbers:
            to_write = '|'.join([cc_number, exp_month, exp_year]) + os.linesep
            fs.write(to_write)


def main():
    card_mask = '427417***2741700'
    exp_year = '2023'
    exp_month = '04'

    candidates = get_candidates(card_mask)

    candidates_file_path = 'cc_candidates.txt'
    write_cc_data(candidates_file_path, candidates, exp_year, exp_month)


if __name__ == '__main__':
    main()
