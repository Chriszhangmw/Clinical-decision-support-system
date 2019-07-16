import pandas as pd
import xlrd

# data_xls = pd.read_excel('./124label.xls',index_col=0)
with open('./124label.csv','r',encoding='utf-8') as f1:
    labeldata = f1.readlines()
print(len(labeldata))
k = 0
with open('./124sym.csv','w',encoding='utf-8') as f2:
    for i in range(len(labeldata)):
        line = list(labeldata[i].replace('\t','').strip())
        print(len(line))
        yan = '0'
        chuxue = '0'
        gengzu = '0'
        chuankong = '0'
        shan = '0'
        ai = '0'
        if line[0] == '1' or line[6] == '1' or line[12] == '1' or line[18] == '1' or line[24] == '1' or line[30] == '1' or line[36] == '1' or line[42] == '1':
            yan = '1'
        if line[1] == '1' or line[7] == '1' or line[13] == '1' or line[19] == '1' or line[25] == '1' or line[31] == '1' or line[37] == '1' or line[43] == '1':
            chuxue = '1'
        if line[2] == '1' or line[8] == '1' or line[14] == '1' or line[20] == '1' or line[26] == '1' or line[32] == '1' or line[38] == '1' or line[44] == '1':
            gengzu = '1'
        if line[3] == '1' or line[9] == '1' or line[15] == '1' or line[21] == '1' or line[27] == '1' or line[33] == '1' or line[39] == '1' or line[45] == '1':
            chuankong = '1'
        if line[4] == '1' or line[10] == '1' or line[16] == '1' or line[22] == '1' or line[28] == '1' or line[34] == '1' or line[40] == '1' or line[46] == '1':
            shan = '1'
        if line[5] == '1' or line[11] == '1' or line[17] == '1' or line[23] == '1' or line[29] == '1' or line[35] == '1' or line[41] == '1' or line[47] == '1':
            ai = '1'
        f2.write(yan + ',' + chuxue +','+gengzu+','+chuankong+','+shan+','+ai +'\n')

