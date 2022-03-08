from io import StringIO
from venv import create


def create_tsv_content(
    lines_cnt: int = 10
) -> StringIO:
    content = '\n'.join([
        f'{line_idx}\t{line_idx*100}'
        for line_idx in range(lines_cnt)
    ])
    tsv_content = StringIO(content)
    return tsv_content


def assess_content(
    content: StringIO,
    separator: str = '\t'
) -> int:
    result = 0

    for line in content:
        col1_value, col2_value = line.split(separator)
        result += int(col1_value) * int(col2_value)

    return result


def main():
    tsv_content = create_tsv_content()
    print( assess_content(tsv_content) )


if __name__ == '__main__':
    main()
