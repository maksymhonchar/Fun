import random


def guessing_game() -> None:
    lower_range = 0
    upper_range = 100
    correct_answer = random.randint(lower_range, upper_range)

    current_attempt = 0
    max_attempts = 3

    while True:
        user_guess_str = input(f'[Attempt {current_attempt + 1}/{max_attempts}] Guess the number: ')

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

        current_attempt += 1
        if current_attempt == max_attempts:
            print('Sorry, you didnt guess in time')
            break


if __name__ == '__main__':
    guessing_game()
