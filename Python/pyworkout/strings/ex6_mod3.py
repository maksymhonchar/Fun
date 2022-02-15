import re


def parse_log(
    log: str
) -> dict:
    parsed_data = {}
    parsed_data_format_helper = [
        ['ip', 'ip address of the client that made the request'],
        ['hyphen', 'identity of the client'],
        ['userid', 'userid of the person requesting the resource'],
        ['dt', 'date and time of the request'],
        ['reqtype_reqresource', 'request type and resource being requested'],
        ['http_code', 'HTTP response status code'],
        ['objsize', 'size of the object returned to the client']
    ]

    pattern = '^((?:[0-9]{1,3}\.){3}[0-9]{1,3})\s(.+)\s(.+)\s(\[.*])\s(\".*\")\s(\d+)\s(\d+)'
    match = re.match(pattern, log)

    for group_idx, [key_name, key_description] in enumerate(parsed_data_format_helper, start=1):
        parsed_data[key_name] = {
            'value': match.group(group_idx),
            'description': key_description
        }

    return parsed_data


def main():
    logs = [
        r'127.0.0.1 - - [28/Jul/2006:10:27:32 -0300] "GET /hidden/ HTTP/1.0" 404 7218',
        r'192.168.2.20 - - [28/Jul/2006:10:27:10 -0300] "GET /cgi-bin/try/ HTTP/1.0" 200 3395',
        r'127.0.0.1 - - [28/Jul/2006:10:22:04 -0300] "GET / HTTP/1.0" 200 2216',
        r'127.0.0.1 - Scott [10/Dec/2019:13:55:36 -0700] "GET /server-status HTTP/1.1" 200 2326'
    ]
    for log in logs:
        result = parse_log(log)
        print( [result_value['value'] for result_value in result.values()] )


if __name__ == '__main__':
    main()
