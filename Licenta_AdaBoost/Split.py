import numpy as np

training_data =[[1., 2., 1.0], [2., 3., 1.0], [3., 4., -1.0], [3., 2., -1.0], [3., 1., -1.0],[4.,4.,-1.0],[5.,4.,-1.0],[5.,2.,1.0],[5.,1.,1.0]]
header = ["x1", "x2"]

def continue_attribute(col, values):
    matrix=np.column_stack((col,values))
    dict_values = dict()
    for x in matrix:
        if x[0] not in dict_values:
            dict_values[x[0]] = set()
        dict_values[x[0]].add(x[1])
    dict_values = sorted(dict_values.items(), key=lambda t: t[0])
    list_attribute = []
    print(dict_values)
    for i in range(len(dict_values) - 1):
        set1 = dict_values[i][1]
        set2 = dict_values[i + 1][1]
        print(dict_values[i][0],set1,dict_values[i+1][0],set2)
        if set1!=set2:
                print(set1.difference(set2))
                list_attribute.append((dict_values[i][0] + dict_values[i + 1][0]) / 2)
        elif len(set1)==2:
            print(set1.difference(set2))
            list_attribute.append((dict_values[i][0] + dict_values[i + 1][0]) / 2)
    print( list_attribute)

def column(matrix, i):
    return [row[i] for row in matrix]

#continue_attribute(column(training_data,0),column(training_data,2))

continue_attribute(column(training_data,1),column(training_data,2))


