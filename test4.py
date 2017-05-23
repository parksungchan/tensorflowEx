# [None, None, None, None, {'TrainResult': [['Trainning ..................................................']
#     , ['Global Step: 1, Training Batch Accuracy: 65.18%, Cost: 1.27595']
#     , ''
#     , ['Trainning ..................................................']
#     , ['Global Step: 2, Training Batch Accuracy: 54.66%, Cost: 1.5225'], '']}
#     , {'labels': ['dog', 'cat', 'motor', 'car', 'glove', 'airplane', 'bolt'], 'predicts': [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 26, 2, 8, 3, 11], [0, 0, 2, 37, 4, 2, 5], [0, 0, 2, 4, 28, 2, 13], [0, 0, 6, 5, 4, 30, 4], [0, 0, 7, 8, 10, 5, 19]]}]
# Tr



a = []
b = []
a.append(b)
print(a)

b.append("aa")
b.append("bb")
b.append("aaa")
print(a)

b = []
b.append("cc")
b.append("dd")
b.append("ccc")
a.append(b)
print(a)
for i in range(len(a)):
    # for j in range(len(a[i])):
    print(i)

for i in a:
    print(i)