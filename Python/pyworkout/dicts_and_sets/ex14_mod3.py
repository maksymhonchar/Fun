import datetime


def ask_age(
    family_dataset: dict
) -> None:
    user_input_name = input('Enter the name of someone in my family: ')

    if user_input_name in family_dataset:
        today_date = datetime.datetime.now().date()
        family_member_bdate = family_dataset[user_input_name]
        age = (today_date - family_member_bdate).days / 365.25
        print(f'[{user_input_name}] is approx {age:.2f} years old')
    else:
        print(f'Cant find family member [{user_input_name}]')


def main():
    family_dataset = {
        'John': datetime.date(1998, 7, 8),
        'Mikhail': datetime.date(2003, 6, 7),
        'Nadine': datetime.date(1994, 6, 14),
    }
    ask_age(family_dataset)


if __name__ == '__main__':
    main()
