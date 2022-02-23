from typing import List as t_List


def get_shell_values(
    content: t_List[str]
) -> t_List[str]:
    separator = ':'
    values = [line.split(separator)[-1] for line in content]
    return values


def get_values_popularity(
    values: t_List[str]
) -> t_List[tuple]:
    popularity_stats = []
    unique_values = set(values)

    for unique_value in unique_values:
        occurences = values.count(unique_value)
        popularity_stats.append((occurences, unique_value))

    return popularity_stats


def main():
    filepath = '/etc/passwd'
    with open(filepath, 'r') as fs_r:
        etc_passwd_content = fs_r.readlines()

    shell_values = get_shell_values(etc_passwd_content)
    shell_values_popularity = get_values_popularity(shell_values)
    sorted_shell_values_popularity = sorted(
        shell_values_popularity, reverse=True
    )

    print(sorted_shell_values_popularity)


if __name__ == '__main__':
    main()
