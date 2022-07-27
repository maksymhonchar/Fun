import random
import time
from typing import Any, Generator, Iterable, Tuple


def timed_generator(
    data: Iterable[Any]
) -> Generator[Tuple[float, Any], None, None]:
    elapsed = 0.0
    for item in data:
        pre_yield_time = time.perf_counter()
        yield elapsed, data
        post_yield_time = time.perf_counter()
        elapsed = post_yield_time - pre_yield_time


def main():
    data = 'abc_hello_def'
    for elapsed, item in timed_generator(data):
        print(f'{elapsed=:.3f} {item=}')
        sleep_secs = random.randint(1, 10) / 10.0
        time.sleep(sleep_secs)


if __name__ == '__main__':
    main()
