import re
from collections import defaultdict


def analyze_heroku_log(
    log_file_path: str
) -> dict:
    analysis = defaultdict(int)
    pattern = '.*status=(\d+).*'

    with open(log_file_path) as fs_r:
        for line in fs_r:
            re_match = re.match(pattern, line)
            if re_match:
                http_code = re_match.group(1)
                analysis[http_code] += 1

    return analysis


def main():
    log_file_path = 'heroku_log.log'
    analysis = analyze_heroku_log(log_file_path)
    print(analysis)


if __name__ == '__main__':
    main()
