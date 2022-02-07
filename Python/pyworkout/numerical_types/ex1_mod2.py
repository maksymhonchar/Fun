import random


def guessing_game() -> None:
    lower_range = 0
    upper_range = 20
    correct_answer = random.randint(lower_range, upper_range)

    number_base_str = input('Enter the number base: ')
    number_base_int = int(number_base_str)

    while True:
        user_guess_str = input(f'[Number base is {number_base_int}] Guess the number: ')

        try:
            user_guess_int = int(user_guess_str, base=number_base_int)
        except ValueError:
            print('User input is not a number')
            continue

        if user_guess_int > correct_answer:
            print('Too high')
        elif user_guess_int < correct_answer:
            print('Too low')
        else:
            print('Just right')
            break


if __name__ == '__main__':
    guessing_game()
