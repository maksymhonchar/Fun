from typing import List as t_List

import pandas as pd


def parse_content(
    content: t_List[str]
) -> pd.DataFrame:
    separator = ':'
    splitted_lines = [line.split(separator) for line in content]
    parsed_content_df = pd.DataFrame(
        [(item[0], item[-1]) for item in splitted_lines],
        columns=['username', 'shell']
    )
    return parsed_content_df


def get_stats(
    content_df: pd.DataFrame
) -> pd.DataFrame:
    stats_df = pd.DataFrame(columns=['popularity', 'usernames'])

    for shell, group in content_df.groupby(by=['shell']):
        stats_df.loc[shell, 'popularity'] = group.shape[0]
        stats_df.loc[shell, 'usernames'] = ','.join( group['username'].unique() )

    stats_df = stats_df.sort_values(by='popularity', ascending=False)

    return stats_df


def main():
    filepath = '/etc/passwd'
    with open(filepath, 'r') as fs_r:
        etc_passwd_content = fs_r.readlines()

    parsed_content_df = parse_content(etc_passwd_content)
    stats = get_stats(parsed_content_df)

    print(stats)


if __name__ == '__main__':
    main()
