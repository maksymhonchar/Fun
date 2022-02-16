def strsort(
    input: str 
) -> str:
    output = ''.join( list(sorted(input, key=str.lower)) )
    return output


def strsort_v2(
    input: str
) -> str:
    input_as_list = list(input.lower())
    output = ''.join( input_as_list.sort() )


def main() -> None:
    user_input = input('Input a word: ')
    print(f'[sorted()]\t{strsort(user_input)}')
    print(f'[list.sort()]\t{strsort(user_input)}')


if __name__ == '__main__':
    main()
