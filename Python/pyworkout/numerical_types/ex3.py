def run_timing() -> None:
    run_time_sum = 0.0
    number_of_runs = 0

    while True:
        run_time_str = input('Enter 10 km run time: ')

        input_is_enter = run_time_str == ''
        if input_is_enter:
            break

        try:
            run_time_float = float(run_time_str)
        except ValueError:
            print(f'Try again: [{run_time_str}] input is not a number')
            continue

        run_time_sum += run_time_float
        number_of_runs += 1

    no_runs_given = number_of_runs == 0
    if no_runs_given:
        avg_run_time = float('nan')
    else:
        avg_run_time = run_time_sum / number_of_runs

    print(f'Run Timing: Average of {avg_run_time:.2f}, over {number_of_runs} runs')


if __name__ == '__main__':
    run_timing()
