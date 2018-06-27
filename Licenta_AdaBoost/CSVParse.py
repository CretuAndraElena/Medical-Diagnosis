import csv


def csv_parse(file_name):
    data = []
    rows = []
    with open('./' + file_name) as csvDataFile:
        csv_reader = csv.reader(csvDataFile)
        for row in csv_reader:
            data.append(row)
    fields = data[0]

    for row in data[1:]:
        new_row = [0] * 25
        for i in range(25):
            if i in (0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17):
                new_row[i] = float(row[i])
            else:
                if i == 24:
                    if row[i] == "ckd":
                        new_row[i] = 1
                    else:
                        new_row[i] = -1
                else:
                    new_row[i] = row[i]
        rows.append(new_row)
    return fields, rows

