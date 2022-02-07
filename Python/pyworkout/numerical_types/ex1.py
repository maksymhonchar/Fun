import random


def guessing_game() -> None:
    lower_range = 0
    upper_range = 100
    correct_answer = random.randint(lower_range, upper_range)

    while True:
        user_guess_str = input('Guess the number: ')

        try:
            user_guess_int = int(user_guess_str)
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
