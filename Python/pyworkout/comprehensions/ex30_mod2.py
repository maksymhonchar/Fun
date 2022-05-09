def create_family() -> dict:
    return {
        'Z': {
            'A': ['B', 'C', 'D'],
            'E': ['F', 'G']
        }
    }


def get_grandchildren(
    family: dict
) -> list:
    """
    Note:
        gen 1: grandmother/grandfather
        gen 2: dad/mom
        gen 3: son/daughter
    """
    return [
        gen3_member
        for gen1_member, gen1_children in family.items()
        for gen2_member, gen2_children in gen1_children.items()
        for gen3_member in gen2_children
    ]


def main():
    family = create_family()
    grandchildren = get_grandchildren(family)
    print(grandchildren)


if __name__ == '__main__':
    main()
