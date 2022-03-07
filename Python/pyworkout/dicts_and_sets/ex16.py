from collections import defaultdict


def dictdiff(
    d1: dict,
    d2: dict
) -> dict:
    result = {}

    for key in d1.keys() | d2.keys():
        d1_value = d1.get(key, None)
        d2_value = d2.get(key, None)

        if d1_value != d2_value:
            result[key] = [d1_value, d2_value]

    return result


def dictdiff_v2(
    d1: dict,
    d2: dict
) -> set:
    d1_items_as_set = set(d1.items())
    d2_items_as_set = set(d2.items())
    diff = d1_items_as_set ^ d2_items_as_set
    return diff


def dictdiff_bad_approach(
    d1: dict,
    d2: dict
) -> dict:
    joined_values = defaultdict(list)

    for key in d1.keys():
        joined_values[key].append( d1.get(key, None) )
        if key not in d2:
            joined_values[key].append(None)

    for key in d2.keys():
        if key not in d1:
            joined_values[key].append(None)
        joined_values[key].append( d2.get(key, None) )

    result = {}
    for key, values in joined_values.items():
        d1_value, d2_value = values
        if d1_value != d2_value:
            result[key] = values

    return result


def main():
    d1 = {'a': 1, 'b': 2, 'c': 3}
    d2 = {'a': 1, 'b': 2, 'c': 4}
    print(dictdiff(d1, d1))
    print(dictdiff(d1, d2))

    d3 = {'a': 1, 'b': 2, 'd': 3}
    d4 = {'a': 1, 'b': 2, 'c': 4}
    print(dictdiff(d3, d4))

    d5 = {'a': 1, 'b': 2, 'd': 4}
    print(dictdiff(d1, d5))


if __name__ == '__main__':
    main()
