import numpy as np
import CSVParse as CSVP
'''training_data = [['big', 'far', 205, -1],
                 ['big', 'near', 205, -1],
                 ['big', 'near', 260, 1],
                 ['big', 'near', 380, 1],
                 ['small', 'far', 205, -1],
                 ['small', 'far', 260, 1],
                 ['small', 'near', 260, 1],
                 ['small', 'near', 380, -1],
                 ['small', 'near', 380, -1]]

header = ["size", "orbit", "temperature", "habitable"]

test_data = [['big', 'near', 280, 1],
             ['big', 'near', 260, 1],
             ['big', 'near', 380, 1],
             ['small', 'far', 205, -1],
             ['small', 'far', 260, 1],
             ['big', 'far', 205, -1],
             ['big', 'near', 205, -1]]
xPrediction=['big', 'near', 280, 1]

training_data = [[1., 2., 1.0], [2., 3., 1.0], [3., 4., -1.0], [3., 2., -1.0], [3., 1., -1.0], [4., 4., -1.0],
                 [5., 4., -1.0], [5., 2., 1.0], [5., 1., 1.0]]
header = ["x1", "x2"]'''

header, training_data = CSVP.csv_parse("DateAndrenament.csv")
header_test, test_data = CSVP.csv_parse("DateTest.csv")
xPrediction=training_data[0]

def continue_attribute(col_number, col, values):
    dict_values = dict()
    for i in range(len(col)):
        if col[i] not in dict_values:
            dict_values[col[i]] = set()
        dict_values[col[i]].add(values[i])
    dict_values = sorted(dict_values.items(), key=lambda t: t[0])
    list_attribute = []
    for i in range(len(dict_values) - 1):
        set1 = dict_values[i][1]
        set2 = dict_values[i + 1][1]
        if set1 != set2:
            list_attribute.append(
                Smaller_Decision_Stump(None, None, col_number, (dict_values[i][0] + dict_values[i + 1][0]) / 2))
            list_attribute.append(
                Bigger_Decision_Stump(None, None, col_number, (dict_values[i][0] + dict_values[i + 1][0]) / 2))
        elif len(set1) == 2:
            list_attribute.append(
                Smaller_Decision_Stump(None, None, col_number, (dict_values[i][0] + dict_values[i + 1][0]) / 2))
            list_attribute.append(
                Bigger_Decision_Stump(None, None, col_number, (dict_values[i][0] + dict_values[i + 1][0]) / 2))
    return list_attribute


class Smaller_Decision_Stump:
    def __init__(self,
                 true_branch,
                 false_branch, column, value):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.false_decision = 0
        self.true_decision = 0

    def evaluation(self, row):
        val = row[self.column]
        if is_numeric(val):
            return self.value<val
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = "<"
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

    def __str__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = "<"
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


class Bigger_Decision_Stump:

    def __init__(self,
                 true_branch,
                 false_branch, column, value):
        self.column = column
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.false_decision = 0
        self.true_decision = 0

    def evaluation(self, row):
        val = row[self.column]
        if is_numeric(val):
            return self.value>=val
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition =">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

    def __str__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


def column(matrix, i):
    return [row[i] for row in matrix]


def partition(rows, attribute):
    true_rows, false_rows = [], []
    i = 0
    for row in rows:
        if attribute.evaluation(row):
            true_rows.append((row, i))
        else:
            false_rows.append((row, i))
        i = i + 1
    return true_rows, false_rows


def decision(rows):
    true = []
    false = []
    for row in rows:
        if row[0][-1] == -1.0:
            false.append(row[1])
        else:
            true.append(row[1])
    if len(true) >= len(false):
        return 1
    else:
        return -1


def count_error(rows, value):
    error = []
    for row in rows:
        if row[0][-1] != value:
            error.append(row[1])
    return error


def weigthed_training_error(D, decision_stump):
    error = 0
    error_rows = []
    true = count_error(decision_stump.true_branch, decision_stump.true_decision)
    false = count_error(decision_stump.false_branch, decision_stump.false_decision)
    error_rows.extend(false)
    error_rows.extend(true)
    for i in error_rows:
        error += D[i]
    return error


def list_of_attribute(rows, h_value):
    list_attribute = []
    for col in range(0, len(rows[0])-1):  # for each column
        values = list(set([row[col] for row in rows]))  # unique values in the column

        if is_numeric(values[0]):
            list_attribute.extend(continue_attribute(col, column(rows, col), column(rows, -1)))
        else:
            list_attribute.append(Smaller_Decision_Stump(None, None, col, values[0]))
    result = []
    for attribute in list_attribute:
        if h_value is None:
            result.append(attribute)
        elif str(h_value) != str(attribute):
            result.append(attribute)

    return result


def find_best_split(rows, D, h_values):
    list_attribute = list_of_attribute(rows, h_values)
    best_attribute = list_attribute[0]
    true_rows, false_rows = partition(rows, best_attribute)
    best_attribute.true_branch = true_rows
    best_attribute.false_branch = false_rows
    best_attribute.true_decision = decision(true_rows)
    best_attribute.false_decision = decision(false_rows)
    min_error = weigthed_training_error(D, best_attribute)
    error = min_error
    for i in range(1, len(list_attribute)):
        attribute = list_attribute[i]
        true_rows, false_rows = partition(rows, attribute)
        attribute.true_branch = true_rows
        attribute.false_branch = false_rows
        attribute.true_decision = decision(true_rows)
        attribute.false_decision = decision(false_rows)

        if len(true_rows) == 0 or len(false_rows) == 0:
            continue
        if attribute.value == list_attribute[i - 1].value and attribute.column == list_attribute[i - 1].column:
            error = 1 - error
        else:
            error = weigthed_training_error(D, attribute)

        if error <= min_error:
            min_error, best_attribute = error, attribute

    return min_error, best_attribute


def build_tree(rows, D, h_value):
    error, attribute = find_best_split(rows, D, h_value)

    return attribute, error


def decistion_stemp(data, D, h_values):
    tree, error = build_tree(data, D, h_values)
    return tree.column, tree, error


def print_tree(node):
    print(str(node.attribute))


def h(h_value, x):
    if is_numeric(x):
        if isinstance(h_value, Smaller_Decision_Stump):
            return np.sign(h_value.value - x)
        else:
            return np.sign(x - h_value.value)
    elif h_value == x:
        return np.sign(1.0)
    else:
        return np.sign(-1.0)


def adaBoost(data, D, h_values):
    col, h_value, error = decistion_stemp(data, D, h_values)
    error=abs(error)
    a = 1 / 2 * np.log((1 - error) / error)
    for i in range(0, len(data)):
        e = np.e ** (-1 * a)
        x = e ** (data[i][-1] * h(h_value, data[i][col]))
        D[i] = D[i] * x
    Z = sum(D)
    D = D / Z
    return (D, error, a, h_value, col)


def predict(a_values, h_values, t, row, col_values):
    prediction = 0
    for j in range(0, t):
        prediction = prediction + a_values[j] * h(h_values[j], row[col_values[j]])
    if prediction < 0:
        return np.sign(-1.0)
    else:
        return np.sign(1.0)


def algorithm(data, t):
    D = [1 / len(data)] * len(data)
    a_values = []
    h_values = []
    col_values = []
    error_values = []
    H = [0] * len(data)
    print(D)
    result = adaBoost(data, D, None)
    D = result[0]
    error_values.append(result[1])
    a_values.append(result[2])
    h_values.append(result[3])
    col_values.append(result[4])

    for i in range(1, t):
        result = adaBoost(data, D, h_values[i - 1])
        D = result[0]
        error_values.append(result[1])
        a_values.append(result[2])
        h_values.append(result[3])
        col_values.append(result[4])

    for i in range(0, len(data)):
        H[i] = predict(a_values, h_values, t, data[i], col_values)
    print("H=", H)
    print("a=", a_values)
    print("h=", h_values)
    print("col=", col_values)
    return (a_values, h_values, col_values)


def error(test_data,data,t):
    a_values,h_values,col_values=algorithm(data,t)
    error=0

    print("Prediction",predict(a_values,h_values,t,xPrediction,col_values))
    for data in test_data:
        if data[-1]!=predict(a_values,h_values,t,data,col_values):
            error=error+1
    print("Error=",error/len(test_data))

error(test_data,training_data,5)

