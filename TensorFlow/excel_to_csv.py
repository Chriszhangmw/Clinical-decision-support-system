

import pandas as pd

data_xls = pd.read_excel('./124label.xls')
with open('./124label.txt','w',encoding='utf-8') as f:
    for line in data_xls:
        f.write(str(line))
# data_xls.to_csv('./124label.txt',encoding='utf-8')