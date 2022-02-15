import re


def remove_names(
    text: str,
    names: list,
    replacement: str = '_'
) -> str:
    fixed_words = []

    words = re.split(r'(\b)', text)
    for word in words:
        word_to_append = replacement if word in names else word
        fixed_words.append(word_to_append)

    fixed_text = ''.join(fixed_words)

    return fixed_text


def main():
    text = """Morning run results:
    1. [Mikhail]
    2. [John]
    3. [Maxim]
    4. [George]
    5. [Ivan]
    """
    names = ['Maxim', 'Ivan']
    replacement = '_deleted_'
    print( remove_names(text, names, replacement) )


if __name__ == '__main__':
    main()
