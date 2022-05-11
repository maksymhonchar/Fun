from collections import defaultdict
import pprint
import json
import urllib.request
from typing import List


def transform_cities_v1(
    data: List[dict]
) -> dict:
    return {
        item['city']: item['population']
        for item in data
    }


def transform_cities_v2(
    data: List[dict]
) -> dict:
    return {
        (item['state'], item['city']): item['population']
        for item in data
    }


def display_differences(
    data: List[dict]
) -> None:
    cities_stats = defaultdict(set)
    for data_item in data:
        cities_stats[data_item['city']].add(data_item['state'])

    cities_with_multiple_states = {
        city: states
        for city, states in cities_stats.items()
        if len(states) >= 2
    }
    pprint.pprint(cities_with_multiple_states)


def main():
    data_url = 'https://gist.githubusercontent.com/reuven/77edbb0292901f35019f17edb9794358/raw/2bf258763cdddd704f8ffd3ea9a3e81d25e2c6f6/cities.json'
    data_str = urllib.request.urlopen(data_url).read().decode('utf-8')
    data_json = json.loads(data_str)

    result_v1 = transform_cities_v1(data_json)
    print(len(result_v1))  # 925

    result_v2 = transform_cities_v2(data_json)
    print(len(result_v2))  # 1000

    display_differences(data_json)


if __name__ == '__main__':
    main()
