from typing import List, Any


def transform(
    data: List[dict]
) -> List[tuple]:
    return [
        (key, value)
        for dict_item in data
        for key, value in dict_item.items()
    ]


def main():
    data = [
        {"key": "value", "123": 123},
        {"another_key": "another_value", "key": "value"},
        {"name": "surname", 123: 123, "hello": "world"}
    ]
    result = transform(data)
    print(result)


if __name__ == "__main__":
    main()
