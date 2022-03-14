import json
import os
from collections import defaultdict


def get_analysis(
    filepath: str
) -> dict:
    with open(filepath, 'r') as fs_r:
        content = json.load(fs_r)

    raw_scores = defaultdict(list)
    for scores_record in content:
        for discipline, score in scores_record.items():
            raw_scores[discipline].append(score)

    analysis = defaultdict(dict)
    for discipline, scores in raw_scores.items():
        analysis[discipline]['min'] = min(scores)
        analysis[discipline]['max'] = max(scores)
        analysis[discipline]['avg'] = sum(scores) / len(scores)

    return analysis


def print_analysis(
    analysis: dict,
    tabulated: bool = True
) -> None:
    tab_char = '\t' if tabulated else ''

    strings_to_display = [
        '{0}{1}: min {2}, max {3}, average {4}'.format(
            tab_char,
            discipline,
            scores_stats['min'], scores_stats['max'], scores_stats['avg']
        )
        for discipline, scores_stats in analysis.items()
    ]

    string_to_display = (os.linesep).join(strings_to_display)
    print(string_to_display)


def print_scores(
    dir_path: str
) -> None:
    try:
        scores_filepaths = os.listdir(dir_path)
    except FileNotFoundError:
        error_msg = f'Cant open directory {dir_path=}'
        raise FileNotFoundError(error_msg)

    for filename in scores_filepaths:
        filepath = os.path.join(dir_path, filename)
        if os.path.isfile(filepath):
            analysis = get_analysis(filepath)
            print(filepath)
            print_analysis(analysis)


def main():
    scores_dir = 'scores'
    print_scores(scores_dir)


if __name__ == '__main__':
    main()
