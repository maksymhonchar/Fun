import statistics
from collections import defaultdict


def get_rainfall() -> str:
    rainfall_data_storage = defaultdict(list)

    while True:
        user_input_city = input('Enter the name of a city: ')

        user_input_city_empty = user_input_city == ''
        if user_input_city_empty:
            break

        user_input_volume = input('Enter rain volume: ')

        try:
            user_input_volume = float(user_input_volume)
        except ValueError:
            print('Rain volume is not a number. Try again')
            continue

        rainfall_data_storage[user_input_city].append(user_input_volume)

    if rainfall_data_storage:
        report = '\n'.join([
            f'{city}: total={sum(volume):.2f}; mean={statistics.mean(volume):.2f}'
            for city, volume in rainfall_data_storage.items()
        ])
    else:
        report = 'Report is empty: no datapoints entered'

    return report


def main():
    rainfall_volume_report = get_rainfall()
    print(rainfall_volume_report)


if __name__ == '__main__':
    main()
