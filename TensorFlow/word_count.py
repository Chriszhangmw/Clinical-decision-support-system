import pandas as pd
import turtle
import jieba.posseg
import re
import xlsxwriter

retrive=('n','v','l','ng','nl','nz','f','vn','a','i','vg','tg','t')
count=11
numbers=[]

words=[]

yScale=15
xScale=36

def drawLine(t,x1,y1,x2,y2):
    t.penup()
    t.goto(x1,y1)
    t.pendown()
    t.goto(x2,y2)

def drawText(t,x,y,text):
    t.penup()
    t.goto(x,y)
    t.pendown()
    t.write(text)

def drawRectangle(t,x,y):
    x = x * xScale
    y = y * yScale
    drawLine(t, x - 8, 0, x - 8, y)
    drawLine(t, x - 8, y, x + 8, y)
    drawLine(t, x + 8, y, x + 8, 0)


def drawBar(t):
    for i in range(count):
        drawRectangle(t,i+1,numbers[i])

def drawGraph(t):
    drawLine(t, 0, 0, 420, 0)
    drawLine(t, 0, 280, 0, 0)
    for x in range(count):
        x = x + 1
        drawText(t, x * xScale - 5, -16, words[x - 1])
        drawText(t, x * xScale - 3, numbers[x - 1] * yScale + 2, numbers[x - 1])
    drawBar(t)

def replacePunctuations(line):
    for ch in line:
        if ch in '~@#$%^()_-+=<>?/,.:;{}[]|""':
            line = line.replace(ch, ' ')
    return line


def cut_line(string):
    string=','.join(re.findall(u'[\u4e00-\u9fff]+', string))
    scut=jieba.posseg.cut(string)
    wordSum=[]
    oneLine=[]
    for word in scut:
        wordSum.append((word.word,word.flag))
    for cuttedword in wordSum:
        if cuttedword[1] in retrive[0:13]:
            tmp=cuttedword[0]+''
            #print(a)
            oneLine.append(tmp)
    return oneLine

def processLine(line,wordCounts):
    line = replacePunctuations(line)
    words = cut_line(line)
    for word in words:
        if word in wordCounts:
            wordCounts[word] += 1
        else:
            wordCounts[word] = 1



def writeExcel(worksheet,job,row=0,name='中文词',number='词频'):
    if row==0:
        worksheet.write(row,0,name)
        worksheet.write(row,1,number)
    else:
        worksheet.write(row,0,job[1])
        worksheet.write(row,1,job[0])


def main():

    workbook = xlsxwriter.Workbook('./temp_samples/statics.xlsx')
    worksheet = workbook.add_worksheet()

    f = pd.read_excel('./temp_samples/final.xlsx', sheet_name=0)
    f = f['出院诊断']


    wordCounts = {}
    for line in f:
        processLine(line.lower(), wordCounts)
    pairs = list(wordCounts.items())
    items = [[x, y] for (y, x) in pairs]
    items.sort()
    for i in range(len(items)-1,0,-1):
        print(items[i][1] + '\t' + str(items[i][0]))
        writeExcel(worksheet,items[i],row=i+1)

        numbers.append(items[i][0])
        words.append(items[i][1])
    worksheet.close()




    # turtle.title('词频结果柱状图')
    # turtle.setup(900, 750, 0, 0)
    # t = turtle.Turtle()
    # t.color('red')
    # t.hideturtle()
    # t.width(2)
    # drawGraph(t)


if __name__ == '__main__':
    main()




















#
# def readfile(path):
#     wordlist = []
#     base = open(path,'r')
#     baseinfo = base.readlines()
#     tagf = open('gggg.txt','r')
#     taginfo = tagf.readlines()
#     for i in taginfo:
#         tags = i.split(' ')
#     for i in baseinfo:
#         words = i.split(' ')
#         for word in words:
#             if word != '\t' and word != '\n' and word != ' ' and word != '':
#                 word = word.replace('\t','')
#                 word = word.replace('\n','')
#                 if word != '':
#                     wordlist.append(word)
#         for x in range(len(tags)):
#             tag = tags[x]
#             for k in range(len(wordlist)):
#                 if tag in wordlist[k]:
