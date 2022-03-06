import random
import datetime


def create_dataset(
    days_to_add: int = 30
) -> dict:
    dataset = {}

    today = datetime.datetime.now().date()
    for day_idx in range(days_to_add):
        date = today - datetime.timedelta(days=day_idx)
        
        date_str = date.strftime("%Y-%m-%d")
        temperature = random.randint(-5*100, 10*100) / 100.0
        
        dataset[date_str] = temperature

    return dataset


def display_temperature(
    dataset: dict
) -> None:
    sorted_date_values = sorted( dataset.keys() )
    min_date_value = min(sorted_date_values)
    max_date_value = max(sorted_date_values)

    requested_date = input(f'[{min_date_value} - {max_date_value}] Enter date: ')

    if requested_date in dataset:
        requested_date_idx = sorted_date_values.index(requested_date)

        if requested_date_idx == 0:
            dates_to_display = sorted_date_values[0:]
        else:
            dates_to_display = sorted_date_values[requested_date_idx-1:]

        dataset_to_display = {
            key: value
            for key, value in dataset.items()
            if key in dates_to_display
        }
        print(dataset_to_display)
    else:
        print('Requested datapoint is missing')


def main():
    dataset = create_dataset(days_to_add=7)
    display_temperature(dataset)


if __name__ == '__main__':
    main()
