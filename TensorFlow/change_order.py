


with open('./liver/liver_train_data.csv','r',encoding = 'utf-8') as f:
    data = f.readlines()
  
with open('./liver/liver_train_label.csv','r',encoding = 'utf-8') as f2:
    data_label = f2.readlines()
      
      
f_in = open('./liver/liver_after_changeorder.csv','w',encoding = 'utf-8')
f_label = open('./liver/liver_label_after_changeorder.csv','w',encoding = 'utf-8')
  
for i in range(len(data)):
    line = data[i].strip()
    f_in.write(line+'\n')
    f_label.write(data_label[i].strip() +'\n')
    line_split = line.split(',')
    one = line_split[0].strip()
    one = one.replace(',','')
    two = line_split[1].strip()
    two = two.replace(',','')
    three = line_split[2].strip()
    three = three.replace(',','')
    line2 = one+ ',' + three + ',' +two
    line3 = two +','+one + ','+ three
    line4 = two +',' + three + ',' + one
    line5 = three + ',' + one +','+ two
    line6 = three+ ',' + two+',' + one

    f_in.write(line2+ '\n')
    f_label.write(data_label[i].strip() +'\n')

    # f_in.write(line3+ '\n')
    # f_label.write(data_label[i].strip() +'\n')

    # f_in.write(line4+ '\n')
    # f_label.write(data_label[i].strip() +'\n')
#     f_in.write(line5+ '\n')
#     f_label.write(data_label[i].strip() +'\n')
    f_in.write(line6+ '\n')
    f_label.write(data_label[i].strip() +'\n')
    


