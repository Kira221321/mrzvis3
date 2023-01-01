from neu_net import main_function


if __name__ == "__main__":
    succession = [1, 2, 4, 8, 16, 32, 64]
    line = 3
    column = 4
    mistake = 0.000001
    pace = 500000
    value_of_alpha = 0.000005
    anticipate = 5
    print(main_function(succession, line, column, mistake, pace, value_of_alpha, anticipate))


