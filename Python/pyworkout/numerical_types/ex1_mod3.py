import random


def guessing_game() -> None:
    answers_storage = ['one', 'two', 'three', 'four', 'five']
    correct_answer = random.choice(answers_storage)
    correct_answer_index = answers_storage.index(correct_answer)

    while True:
        user_guess = input('Guess the word: ')

        try:
            user_guess_index = answers_storage.index(user_guess)
        except ValueError:
            print('User input is not in an answers storage')
            continue

        print(user_guess_index, correct_answer_index)

        if user_guess_index > correct_answer_index:
            print('Too high: try earlier word')
        elif user_guess_index < correct_answer_index:
            print('Too low: try later word')
        else:
            print('Just right')
            break


if __name__ == '__main__':
    guessing_game()
