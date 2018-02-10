import pandas as pd,csv

with open("result.csv") as cfile:
    reader = csv.DictReader(cfile)
    f = open("result.txt", "w")
    for row in reader:
        print(row.get('date'))
        f.write(row.get('date') +'\t'+ row.get('value').split('.')[0]+'\n')
    f.close()