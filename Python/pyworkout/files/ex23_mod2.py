import csv
import pprint


def etcpasswd_to_json(
    filepath: str
) -> list:
    etcpasswd_item_keys = [
        'username',
        'password',
        'uid',
        'gid',
        'gecos',
        'home',
        'shell'
    ]
    etcpasswd_items = []

    with open(filepath, 'r') as fs_r:
        reader = csv.reader(fs_r, delimiter=':')
        for line in reader:
            etcpasswd_item = {
                key: value
                for key, value in zip(etcpasswd_item_keys, line)
            }
            etcpasswd_items.append(etcpasswd_item)
    
    return etcpasswd_items


def main():
    filepath = 'etcpasswd'
    etcpasswd_as_dict = etcpasswd_to_json(filepath)
    pprint.pprint(etcpasswd_as_dict)


if __name__ == '__main__':
    main()
