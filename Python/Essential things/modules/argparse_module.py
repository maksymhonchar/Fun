import argparse

def basic_parser_usage():
    parser = argparse.ArgumentParser(
        description='Basic argument parser',
        epilog='python file.py'
    )

    parser.print_help()  # full info: descr, optional args, usage.
    parser.print_usage()  # only 'usage: ...'

def get_args():
    parser = argparse.ArgumentParser(
        description='Second args parser',
        epilog='python file.py [arg1, ]'
    )
    parser.add_argument('-x', action='store', required=True, help='something for option X')
    parser.add_argument('-a', '--all', action='store', required=True, help='smt for option A')
    parser.add_argument('-y', help='option Y descr', default=False)  # type-str
    parser.add_argument('-z', help='option Z descr', type=int)  # default=None
    print(parser.parse_args())

if __name__ == '__main__':
    get_args()
