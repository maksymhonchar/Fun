from typing import Any, Iterator, Sequence


def circle(
    sequence: Sequence[Any],
    max_iterations: int
) -> Iterator:
    return (
        sequence[index % len(sequence)]
        for index in range(max_iterations)
    )


def main():
    circle_generator = circle(
        sequence='abcd',
        max_iterations=10
    )

    print(
        type(circle_generator)
    )

    for item in circle_generator:
        print(item)


if __name__ == '__main__':
    main()
