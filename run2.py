from MyModels import *

import pickle
import pandas as pd
import torch
import io
import os
# from google.colab import files
from tqdm import trange
import ast
import pickle
from sentence_transformers import SentenceTransformer

def show_emoji(index):
    if index < 0 or index > 19:
        print("Invalid index")
        exit(1)

    file_name = 'emoji_'
    file_name += str(index)
    file_name += '.png'

    img = mpimg.imread(file_name)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


df = pd.read_csv("data/emoji_cleaned.csv")

model0 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

_temp = df.Tweet.tolist()
embeddings = model0.encode(df.Tweet.tolist())

print(embeddings.shape)


# hyper-parameters
input_size = embeddings.shape[1]
hidden_size = 256
num_classes = 20
learning_rate = 0.001
batch_size = 16
num_epochs = 3

total_num = embeddings.shape[0]
train_end_num = int(total_num * 0.8)

train_embeddings = embeddings[:train_end_num]
test_embeddings = embeddings[train_end_num+1:]

data = df.values.tolist() # get list
label_list_total = [data[i][1] for i in range(len(data))]

label_list = label_list_total[:train_end_num]
label_list_test = label_list_total[train_end_num+1:]

# 1. to tensor
x = torch.tensor(train_embeddings)
x_test = torch.tensor(test_embeddings)
y = torch.tensor(label_list)
y_test = torch.tensor(label_list_test)

# 2. to TensorDataset
train_dataset = Data.TensorDataset(x, y)
test_dataset = Data.TensorDataset(x_test, y_test)

# 3. Dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = myNN(input_size, hidden_size, num_classes).to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

iter = []
epoch_list = []
loss_log = []
acc_list = []
acc_list_test = []

# train network
iteration = 0
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)

        # print("epoch:", epoch, ", loss:", float(loss))

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        # record info
        iteration += 1
        iter.append(iteration)
        loss_log.append(loss)

    epoch_list.append(epoch)
    accuracy1 = check_accuracy(train_loader, model)
    accuracy2 = check_accuracy(test_loader, model)
    print("epoch:", epoch, ", train accuracy:", accuracy1, ", test accuracy:", accuracy2)
    acc_list.append(accuracy1)
    acc_list_test.append(accuracy2)


# plt.xlabel("iteration")
# plt.ylabel("acc")
# plt.plot(epoch_list, acc_list, color='skyblue', label='test')
# plt.plot(epoch_list, acc_list_test, color='green', label='train')
# plt.show()

accuracy1 = check_accuracy(train_loader, model)
accuracy2 = check_accuracy(test_loader, model)
print("train accuracy: ", accuracy1)
print("test accuracy: ", accuracy2)

# get into shell
while True:
    str_input = input(' > ')
    str_l = [str_input]
    encoding = model0.encode(str_l)
    encoding = torch.tensor(encoding).to(device=device)
    res = model(encoding)
    label = res.argmax(dim=1)
    label = int(label)
    # print(label)
    show_emoji(label)


print("end")