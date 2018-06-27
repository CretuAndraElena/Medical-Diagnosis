import math
import CSVParse as CSVP

header, training_data = CSVP.csv_parse("DateAndrenament.csv")

def unique_vals(rows, col):  # valorile unice de pe o coloana
    return set([row[col] for row in rows])


def label_counts(rows):
    counts = {}  # a dictionary of class -> count.
    for row in rows:
        # class is in the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)

class Leaf:
    def __init__(self, rows):
        self.predictions = label_counts(rows)

    def __str__(self):
        return str(self.predictions)


class Decision_Node:
    def __init__(self,
                 attribute,
                 true_branch,
                 false_branch):
        self.attribute = attribute
        self.true_branch = true_branch
        self.false_branch = false_branch
    def __str__(self):
        return str(self.attribute)

class Attribute:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def evaluation(self, row):
        val = row[self.column]
        if is_numeric(val):
            return val <= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = "<="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

    def __str__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


class ID3:
    def __init__(self, header, data, test_data):
        self.header = header
        self.data = data
        self.test_data = test_data
        self.H_0 = self.entropy(self.data)
        self.tree = self.build_tree(self.data)

    def partition(self, rows, attribute):
        true_rows, false_rows = [], []
        for row in rows:
            if attribute.evaluation(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows

    def entropy(self, rows):
        counts = label_counts(rows)  # dict with 1 or 2 elements
        H = 0
        for i in counts:
            H += (counts[i] / float(len(rows)) * math.log(float(len(rows) / counts[i]), 2))
        return H

    def mutual_info(self, left, right, H_0):
        # H(habitat/Size)=35/80*H(size=x)+45/80*H(size=y)
        # IG=H_0-H(habitat/size)
        p = float(len(left)) / (len(left) + len(right))
        return H_0 - p * self.entropy(left) - (1 - p) * self.entropy(right)

    def continue_attribute(self, col, values, rows):
        dict_values = dict()
        for row in rows:
            for value in values:
                if row[col] == value:
                    if value not in dict_values:
                        dict_values[value] = set()
                    dict_values[value].add(row[-1])
        dict_values = sorted(dict_values.items(), key=lambda t: t[0])
        list_attribute = []
        for i in range(len(dict_values) - 1):
            set1 = dict_values[i][1]
            set2 = dict_values[i + 1][1]
            if len(set1) == 2 and len(set2) == 2:
                if set1.difference(set2) is not set():
                    list_attribute.append(Attribute(col, (dict_values[i][0] + dict_values[i + 1][0]) / 2))
                else:
                    continue
            elif len(set1) == 2 or len(set2) == 2:
                list_attribute.append(Attribute(col, (dict_values[i][0] + dict_values[i + 1][0]) / 2))
            elif set1.difference(set2) is not set():
                list_attribute.append(Attribute(col, (dict_values[i][0] + dict_values[i + 1][0]) / 2))
            else:
                continue
        return list_attribute

    def constant_atribute(self, col, values):
        list_attribute = []
        for value in values:
            list_attribute.append(Attribute(col, value))
        return list_attribute

    def find_best_split(self, rows):
        best_ig = 0  # keep track of the best information gain
        best_attribute = None  # keep train of the feature / value that produced it
        n_features = len(rows[0]) - 1  # number of columns
        list_atribute = []
        for col in range(n_features):  # for each column
            values = list(unique_vals(rows, col))  # unique values in the column
            if is_numeric(values[0]):
                list_atribute.extend(self.continue_attribute(col, values, rows))
            else:
                list_atribute.append(Attribute(col, values[0]))

        for attribute in list_atribute:  # for each value
            # try splitting the dataset
            true_rows, false_rows = self.partition(rows, attribute)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            ig = self.mutual_info(true_rows, false_rows, self.H_0)

            if ig >= best_ig:
                best_ig, best_attribute = ig, attribute
        return best_ig, best_attribute

    def build_tree(self, rows):

        if self.entropy(rows) == 0:
            return Leaf(rows)

        gain, attribute = self.find_best_split(rows)

        true_rows, false_rows = self.partition(rows, attribute)
        true_branch = self.build_tree(true_rows)
        false_branch = self.build_tree(false_rows)
        return Decision_Node(attribute, true_branch, false_branch)

    def print_tree(self, node, spacing=""):

        if isinstance(node, Leaf):
            print(spacing + "Predict", node.predictions)
            return

        print(spacing + str(node.attribute))

        print(spacing + '--> True:')
        self.print_tree(node.true_branch, spacing + "  ")

        print(spacing + '--> False:')
        self.print_tree(node.false_branch, spacing + "  ")

    def classify(self, row, node):
        if isinstance(node, Leaf):
            return node.predictions
        if node.attribute.evaluation(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)

    def print_leaf(self):
        """A nicer way to print the predictions at a leaf."""
        total = sum(self.values()) * 1.0
        probs = {}
        for lbl in self.keys():
            probs[lbl] = str(int(self[lbl] / total * 100)) + "%"
        return probs

    def calculate_error(self):
        error = 0;
        for row in self.test_data:
            if row[-1] not in self.classify(row, self.tree).keys() or (
                    row[-1] in self.classify(row, self.tree).keys() and len(self.classify(row, self.tree)) > 1):
                error = error + 1;
        return error / len(self.test_data)


id3 = ID3(header, training_data, training_data)
print("Eroarea la antrenare:", id3.calculate_error())

header_test, test_data = CSVP.csv_parse("DataSetTest3.csv")
id3 = ID3(header, training_data, test_data)
print("Eroarea la testare:", id3.calculate_error())

def studiu_de_caz():
    header,data_set=CSVP.csv_parse("DataSetStudiuDeCaz.csv")
    result=[]
    for x in data_set:
        result.append(list(id3.classify(x,id3.tree).items())[0][0])
    print("Rezultate studiu de caz:",result)


studiu_de_caz()

import time
start_time = time.time()
id3 = ID3(header, training_data, test_data)
print("Timpul de antrenare:" ,(time.time() - start_time))
