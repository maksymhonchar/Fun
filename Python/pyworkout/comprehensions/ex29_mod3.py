from typing import List
import pprint


def update_dataset(
    people: List[dict],
    adult_age: int = 20
) -> List[dict]:
    return [
        {
            "name": person["name"],
            "age": person["age"],
            "approx_age_in_months": person["age"] * 12
        }
        for person in people
        if person["age"] <= adult_age
    ]


def main():
    people = [
        {"name": "max", "age": 30},
        {"name": "ivan", "age": 81},
        {"name": "alex", "age": 18},
        {"name": "josh", "age": 63},
        {"name": "jake", "age": 19},
    ]
    result = update_dataset(people)
    pprint.pprint(result)


if __name__ == "__main__":
    main()
