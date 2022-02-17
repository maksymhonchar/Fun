from typing import Sequence as t_Sequence


def firstlast(
    sequence
) -> t_Sequence:
    if len(sequence) == 0:
        raise ValueError('firstlast behavior is not defined for empty sequence')
    elif len(sequence) == 1:
        raise ValueError('firstlast behavior is not defined for sequences with 1 item')

    last_element_idx = len(sequence)-1
    result = sequence[::last_element_idx]
    return result


def firstlast_v2(
    sequence
) -> t_Sequence:
    if len(sequence) == 0:
        raise ValueError('firstlast behavior is not defined for empty sequence')
    elif len(sequence) == 1:
        raise ValueError('firstlast behavior is not defined for sequences with 1 item')

    result = sequence[:1] + sequence[-1:]
    return result


def main() -> None:
    func_to_use = [firstlast, firstlast_v2]
    for func in func_to_use:
        print( func( sequence=[1, 2, 3, 4, 5] ) )
        print( func( sequence='a_hello_world_z' ) )
        print( func( sequence=(999, 7, 8, 9, 10, 999) ) )


if __name__ == '__main__':
    main()
