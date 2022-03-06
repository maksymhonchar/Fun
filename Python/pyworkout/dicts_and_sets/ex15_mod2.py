import re
from collections import defaultdict


def analyze_logfile(
    logfile: str
) -> str:
    analysis_data = defaultdict(set)

    pattern = r'^((?:[0-9]{1,3}\.){3}[0-9]{1,3})\s(.+)\s(.+)\s(\[.*])\s(\".*\")\s(\d+)(.*)$'

    for log_line in logfile.split('\n'):
        if log_line:
            match = re.match(pattern, log_line)
            ip_address = match.group(1)
            req_method = match.group(5).split(' ')[0][1:]
            analysis_data[req_method].add(ip_address)

    return analysis_data


def main():
    logfile = \
"""
192.168.198.92 - - [22/Dec/2002:23:08:37 -0400] "GET / HTTP/1.1" 200 6394 www.yahoo.com "-" "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1...)" "-"
192.168.198.92 - - [22/Dec/2002:23:08:38 -0400] "GET /images/logo.gif HTTP/1.1" 200 807 www.yahoo.com "http://www.some.com/" "Mozilla/4.0 (compatible; MSIE 6...)" "-"
192.168.72.177 - - [22/Dec/2002:23:32:14 -0400] "GET /news/sports.html HTTP/1.1" 200 3500 www.yahoo.com "http://www.some.com/" "Mozilla/4.0 (compatible; MSIE ...)" "-"
192.168.72.177 - - [22/Dec/2002:23:32:14 -0400] "GET /favicon.ico HTTP/1.1" 404 1997 www.yahoo.com "-" "Mozilla/5.0 (Windows; U; Windows NT 5.1; rv:1.7.3)..." "-"
192.168.72.177 - - [22/Dec/2002:23:32:15 -0400] "GET /style.css HTTP/1.1" 200 4138 www.yahoo.com "http://www.yahoo.com/index.html" "Mozilla/5.0 (Windows..." "-"
192.168.72.177 - - [22/Dec/2002:23:32:16 -0400] "GET /js/ads.js HTTP/1.1" 200 10229 www.yahoo.com "http://www.search.com/index.html" "Mozilla/5.0 (Windows..." "-"
192.168.72.177 - - [22/Dec/2002:23:32:19 -0400] "GET /search.php HTTP/1.1" 400 1997 www.yahoo.com "-" "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; ...)" "-"
"""
    analysis_data = analyze_logfile(logfile)
    print(analysis_data)


if __name__ == '__main__':
    main()
