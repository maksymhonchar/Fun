def income_tax(
    income: float
) -> float:
    if 0.0 < income <= 1_000.0:
        rate = 0.0
    elif 1_000.0 < income <= 10_000.0:
        rate = 0.1
    elif 10_000.0 < income <= 20_000.0:
        rate = 0.2
    elif income > 20_000.0:
        rate = 0.5
    else:
        error_msg = f"Invalid income argument value [{income=}]"
        raise ValueError(error_msg)

    tax = income * rate
    return tax


def main():
    income = 55_000.55
    tax = income_tax(income)
    print(f'Results: [{income=}] [{tax=}]')


if __name__ == "__main__":
    main()
