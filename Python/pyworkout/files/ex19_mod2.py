from collections import defaultdict


def analyze_integers() -> dict:
    analysis = defaultdict(list)

    user_input = input('Enter integers, separated by spaces: ')
    user_integers = [int(value) for value in user_input.split(' ')]

    for factor in range(1, max(user_integers) + 1):
        for user_integer in user_integers:
            if user_integer % factor == 0:
                analysis[factor].append(user_integer)

    return analysis


def main():
    analysis = analyze_integers()
    print(analysis)


if __name__ == '__main__':
    main()
