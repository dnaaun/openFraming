import csv
import os

# final = []
# with open('covid19-ko.csv') as file:
#     csv_reader = csv.reader(file)
#     for row in csv_reader:
#         final.append(row)

# final2 = []
# final2.append(['date', 'name', 'category', 'value'])
# header = final[0][3:]
# for row in final[1:-1]:
#     for i in range(len(header)):
#         temp_date = header[i].split('-')[0].strip(' ')
#         date = '2020-' + temp_date.split('/')[0] + '-' + temp_date.split('/')[1]
#         temp = [date, row[1], row[2] if row[2]!='' else row[1].split(' ')[0], float(row[i + 3][:-1]) if row[i + 3]!='' else 0]
#         final2.append(temp)
# with open('covid19_KO.csv', 'w+', newline='') as file:
#     csv_writer = csv.writer(file)
#     for ele in final2:
#         csv_writer.writerow(ele)
dic = {
    0: '2nd Amendment',
1: 'Gun control/regulation',
2: 'Politics',
3: 'Mental health',
4: 'School/Public space safety',
5: 'Race/Ethnicity',
6: 'Public opinion',
7: 'Society/Culture',
8: 'Economic consequences'
}
data_2016, data_2017, data_2018 = [], [], []
counter = 0
with open('english2016Results.txt', encoding='utf-8') as file:
    temp = file.read().split('\n')
    for line in temp:
        line_te = line.split('\t')
        try :
            url = line_te[1].strip('http://').strip('https://').split('.')
            date = line_te[0].split(' ')[0].split('/')
            final_date = date[0] + '-' + date[1] + '-20' + date[2]
            final = [final_date, url[1] if url[0]=='www' else url[0], line_te[2], line_te[3], dic[int(line_te[4])]]
            data_2016.append(final)
        except IndexError:
            continue

with open('english2017Results.txt', encoding='utf-8') as file:
    temp = file.read().split('\n')
    for line in temp:
        line_te = line.split('\t')
        try :
            url = line_te[1].strip('http://').strip('https://').split('.')
            date = line_te[0].split(' ')[0].split('/')
            final_date = date[0] + '-' + date[1] + '-20' + date[2]
            final = [final_date, url[1] if url[0]=='www' else url[0], line_te[2], line_te[3], dic[int(line_te[4])]]
            data_2016.append(final)
        except IndexError:
            continue

with open('english2018Results.txt', encoding='utf-8') as file:
    temp = file.read().split('\n')
    for line in temp:
        line_te = line.split('\t')
        try :
            url = line_te[1].strip('http://').strip('https://').split('.')
            date = line_te[0].split(' ')[0].split('/')
            final_date = date[0] + '-' + date[1] + '-20' + date[2]
            final = [final_date, url[1] if url[0]=='www' else url[0], line_te[2], line_te[3], dic[int(line_te[4])]]
            data_2016.append(final)
        except IndexError:
            continue

with open('gunviolence_data.csv', 'w+', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    for line in data_2016:
        writer.writerow(line)
