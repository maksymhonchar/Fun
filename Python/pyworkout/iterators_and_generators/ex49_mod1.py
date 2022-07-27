import random
import time
from typing import Any, Generator, Iterable, Tuple


def elapsed_since(
    data: Iterable[Any],
    min_time: float
) -> Generator[Tuple[float, Any], None, None]:
    last_run_time = None
    for item in data:
        current_time = time.perf_counter()
        elapsed = current_time - (last_run_time or current_time)
        if elapsed < min_time:
            print('add additional wait')
            time.sleep(min_time - elapsed)
        last_run_time = time.perf_counter()
        yield elapsed, item


def main():
    data = '12345'
    min_time = 0.5
    for elapsed, item in elapsed_since(data, min_time):
        time.sleep(random.randint(1, 10) / 10.0)
        print(f'{elapsed=} {item=}')


if __name__ == '__main__':
    main()
