import numpy


def main_function(succession: list, line: int, column: int, mistake: float, pace: int, value_of_alpha: float, anticipate: int):
    x_create = construct(succession, line)
    y_create = construct(succession, line)
    length_x = len(x_create)
    matrix = numpy.zeros(length_x, column)
    x_create = numpy.append(x_create, matrix, axis=1)
    line_col_sum = line+column
    first_weight = numpy.random.rand(line_col_sum, column)
    second_weight = numpy.random.rand(column, 1)
    first_weight = educate(mistake, pace, x_create, first_weight, second_weight, y_create, value_of_alpha)
    second_weight = educate(mistake, pace, x_create, first_weight, second_weight, y_create, value_of_alpha)
    final_result = prognosticate(first_weight, second_weight, anticipate, column, x_create, y_create)
    print(str(f'Последовательность:{succession}'))
    return final_result


def construct(succession: list, line: int):
    x_create, y_create = [], []
    p = 0
    length_of_seq = len(succession)
    while length_of_seq > p + line:
        k = []
        for m in range(line):
            k.append(succession[m + p])
        x_create.append(k)
        y_create.append(succession[p + line])
        p = p + 1 
    x_create = numpy.array(x_create)
    y_create = numpy.array(y_create)
    return x_create, y_create


def operate_func(x_create):
    value_x = len(x_create[0])
    for p in range(value_x):
        x_create[0][p] = numpy.log(x_create[0][p] + (x_create[0][p]**2 + 1)**(0.5))
    return x_create


def derivate_func(x_create):
    value_x = len(x_create[0])
    for p in range(value_x):
        x_create[0][p] = 1/((x_create[0][p] ** 2 + 1)**(0.5))
    return x_create


def educate(mistake, pace, x_create, first_weight, second_weight, y_create, value_of_alpha):
    blunder = 1
    iteration = 1
    while iteration <= pace and mistake <= blunder:
        blunder = 0
        length_of_x = len(x_create)
        for p in range(length_of_x):
            num_x = len(x_create[p])
            num_zer = numpy.zeros((1, num_x))
            for i in range(num_x):
                num_zer[0][i] = x_create[p][i]
            mtrx = operate_func(num_zer @ first_weight)
            final_result = operate_func(mtrx @ second_weight)
            first_value = value_of_alpha * (final_result - y_create[p])
            second_value = num_zer.T @ second_weight.T
            first_weight -= first_value * second_value * derivate_func(num_zer @ first_weight)
            second_weight -= first_value * mtrx.T * derivate_func(mtrx @ second_weight)
            blunder += ((final_result - y_create[p])**2)[0]/2
        print(f'Iteration{iteration}: {blunder}')
        iteration = iteration + 1
    return first_weight, second_weight


def prognosticate(first_weight, second_weight, anticipate, column, x_create, y_create):
    shaped_relation = y_create[-1].reshape(1)
    Z = x_create[-1: column]
    final_result = []
    for p in range(anticipate):
        Z = Z[1:]
        teach = numpy.concatenate((Z, shaped_relation))
        Z = numpy.concatenate((Z, shaped_relation))
        num_arr = numpy.array([0] * column)
        teach = numpy.append(teach, num_arr)
        rez = teach @ first_weight @ second_weight
        shaped_relation = rez
        final_result.append(rez[0])
    return final_result




