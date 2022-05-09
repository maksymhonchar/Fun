def create_family() -> dict:
    return {
        "Z": {
            "A": [
                {"name": "B", "age": 39},
                {"name": "C", "age": 52},
                {"name": "D", "age": 66}
            ],
            "E": [
                {"name": "F", "age": 28},
                {"name": "G", "age": 72}
            ]
        }
    }


def get_sorted_grandchildren(
    family: dict
) -> list:
    """
    Note:
        gen 1: grandmother/grandfather
        gen 2: dad/mom
        gen 3: son/daughter
    """
    return sorted(
        [
            gen3_member
            for gen1_member, gen1_children in family.items()
            for gen2_member, gen2_children in gen1_children.items()
            for gen3_member in gen2_children
        ],
        key=lambda item: item["age"],
        reverse=True
    )


def main():
    family = create_family()
    grandchildren = get_sorted_grandchildren(family)
    print(grandchildren)


if __name__ == "__main__":
    main()
