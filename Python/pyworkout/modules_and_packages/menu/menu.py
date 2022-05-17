from typing import Callable


def error_handler():
    error_msg = "Can't find method similar to user input"
    raise ValueError(error_msg)


def menu(
    **kwargs: Callable
) -> None:
    user_msg = "Enter some input:"
    user_input = input(user_msg)

    func_to_call = kwargs.get(user_input, error_handler)
    result = func_to_call()
    return result


if __name__ == '__main__':
    # run tests in case one invokes the file as a stand-alone program from the cmd
    print('Running dummy test #1...')
    assert(1 == 1)

    print('Running test #2...')
    assert(2 == 2)

    print('Running dummy test #1...')
    assert(3 == 6)
