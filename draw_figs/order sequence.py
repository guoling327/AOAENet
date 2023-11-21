import matplotlib.pyplot as plt


total_width, n = 0.6, 6
width = total_width / n



name_list = ['Cora','Citeseer','Cornell', 'Wisconsin','Texas']
num_list1 = [76.40, 73.79,	85.41, 	86.67, 88.95]
num_list2 = [75.90, 73.46, 	86.22,	88.04, 89.47]
num_list3 = [76.12,	73.68, 87.03, 	85.88, 87.89]
num_list4 = [75.92, 73.28,  85.68,	84.31, 87.89]
num_list5 = [75.27,	73.19, 86.22, 	86.47, 88.95]
num_list6 = [75.79, 73.64,	85.68, 	86.86, 89.47]


# std_err1=[2.10,6.53,4.79]
# std_err2=[5,1,4,2,5]
x = list(range(len(num_list1)))
#plt.figure(figsize=(6.5,6.5))
plt.bar(x, num_list1, width=width, label='3 2 1 0 (original)')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list2, width=width, label='0 1 2 3')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list3, width=width, label='3 2 0 1', tick_label=name_list)
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list4, width=width, label='3 1 0 2')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list5, width=width, label='2 1 0 3')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list6, width=width, label='1 3 2 0')
for i in range(len(x)):
    x[i] = x[i] + width
plt.legend(title='The order sequence')



plt.xticks(rotation=15)
plt.yticks()
#plt.xlabel('Datasets')
plt.ylabel('Accuracy')
plt.ylim(70,90)
plt.savefig('Order.png',dpi=600)
plt.show()
