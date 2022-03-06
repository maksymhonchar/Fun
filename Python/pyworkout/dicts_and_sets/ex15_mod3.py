import re
from collections import defaultdict

import lorem


def analyze_text(
    text: str
) -> dict:
    analysis = defaultdict(int)

    for word in re.split('\W+', text):
        analysis[len(word)] += 1

    return analysis


def main():
    text = lorem.paragraph()
    text_analysis = analyze_text(text)
    print(text_analysis)


if __name__ == '__main__':
    main()
