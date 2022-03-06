def restaurant(
    menu: dict
) -> None:
    total_amount = 0.0

    menu_str = ', '.join(menu)
    user_input_msg = '[Menu: {0}] [Total: {1}] Enter name of a dish: '
    item_missing_msg = 'Sorry, [{0}] is not on the menu. Try something else'
    total_amount_msg = 'Your total will be: [{0}] UAH'

    while True:
        requested_menu_item = input( user_input_msg.format(menu_str, total_amount) )
        requested_menu_item = requested_menu_item.strip()

        requested_menu_item_empty = requested_menu_item == ''
        if requested_menu_item_empty:
            break

        if requested_menu_item in menu:
            total_amount += menu[requested_menu_item]
        else:
            print( item_missing_msg.format(requested_menu_item) )

    print( total_amount_msg.format(total_amount) )


def main():
    menu = {
        'dish_1': 100.50,
        'dish_2': 200.25,
        'drink_1': 300.95,
        'drink_2': 400.50
    }
    restaurant(menu)


if __name__ == '__main__':
    main()
