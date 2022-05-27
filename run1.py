from MyModels import *


model_option = 5

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device in main:", device)

# convert sentences to indices
def convert_to_idx(sents, word2idx):
    for sent in sents:
        for i in range(len(sent)):
            if sent[i] in word2idx.keys():
                sent[i] = word2idx[sent[i]]
            else:
                sent[i] = 0

def label_to_vector(label_list, dim):
    result = []
    for label in label_list:
        temp = [0] * dim
        temp[label] = 1
        result.append(temp)

    return result

# process data
# load word pretrained word embedding
wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/home/yuan/Desktop/w2v.bin"), binary=True)
vocab2idx = wv_from_bin.key_to_index # get word-idx dictionary

# initializing hyper-parameters
# RNN
if model_option == 1:
    input_size = wv_from_bin.vectors.shape[1]
    num_layers = 2
    hidden_size = 128
    num_classes = 20
    learning_rate = 0.007
    batch_size = 16
    num_epochs = 50
# GRU
elif model_option == 2:
    input_size = wv_from_bin.vectors.shape[1]
    num_layers = 2
    hidden_size = 256
    num_classes = 20
    learning_rate = 0.009
    batch_size = 64
    num_epochs = 50
# LSTM
elif model_option == 3:
    input_size = wv_from_bin.vectors.shape[1]
    num_layers = 2
    hidden_size = 256
    num_classes = 20
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 50
# LSTM (capturing all hidden states)
elif model_option == 4:
    input_size = wv_from_bin.vectors.shape[1]
    num_layers = 2
    hidden_size = 256
    num_classes = 20
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 50
# Bi-LSTM
elif model_option == 5:
    input_size = wv_from_bin.vectors.shape[1]
    num_layers = 2
    hidden_size = 256
    num_classes = 20
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 50

# 0. Get list from csv
df = pd.read_csv("data/train_emoji_cleaned.csv")
df_test = pd.read_csv("data/test_emoji_cleaned.csv")
data = df.values.tolist() # get list
data_test = df_test.values.tolist() # get list

# remove 'nan'
maxLen = len(data)
i = 0
while i < maxLen:
    if not isinstance(data[i][0], str):
        del data[i]
        maxLen -= 1
        continue
    i += 1

label_list = [data[i][1] for i in range(len(data))]
label_list_test = [data_test[i][1] for i in range(len(data_test))]
# label_list = label_to_vector(label_list, num_classes)
# y = torch.FloatTensor(label_list)

sent_tok_list = [data[i][0].split() for i in range(len(data))]
sent_tok_list_test = [data_test[i][0].split() for i in range(len(data_test))]

idx_list = copy.deepcopy(sent_tok_list)
idx_list_test = copy.deepcopy(sent_tok_list_test)
convert_to_idx(idx_list, vocab2idx) # getting index list
convert_to_idx(idx_list_test, vocab2idx) # getting index list

# padding embeddings with 0
# get the max length
maxLen = 0
for emb in idx_list:
    if len(emb) > maxLen:
        maxLen = len(emb)
for emb in idx_list_test:
    if len(emb) > maxLen:
        maxLen = len(emb)

for idx, emb in enumerate(idx_list):
    temp = [0] * (maxLen - len(emb))
    idx_list[idx] = emb + temp

for idx, emb in enumerate(idx_list_test):
    temp = [0] * (maxLen - len(emb))
    idx_list_test[idx] = emb + temp

# get embedding list
embedding_list = copy.deepcopy(idx_list)
for idx, word_idx in enumerate(embedding_list):
    embedding_list[idx] = torch.from_numpy(wv_from_bin.vectors[word_idx])

embedding_list_test = copy.deepcopy(idx_list_test)
for idx, word_idx in enumerate(embedding_list_test):
    embedding_list_test[idx] = torch.from_numpy(wv_from_bin.vectors[word_idx])

# 1. to tensor
x = torch.stack(embedding_list, dim=0)
x_test = torch.stack(embedding_list_test, dim=0)
y = torch.tensor(label_list)
y_test = torch.tensor(label_list_test)
# y = torch.FloatTensor(label_list)

# 2. to TensorDataset
train_dataset = Data.TensorDataset(x, y)
test_dataset = Data.TensorDataset(x_test, y_test)

# 3. Dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# initialize network
if model_option == 1:
    model = myRNN(input_size, hidden_size, num_layers, num_classes).to(device)
if model_option == 2:
    model = myGRU(input_size, hidden_size, num_layers, num_classes).to(device)
if model_option == 3:
    model = myLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
if model_option == 4:
    model = myLSTMall(input_size, hidden_size, num_layers, num_classes, sequence_length=maxLen).to(device)
if model_option == 5:
    model = myBiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.1)

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


plt.xlabel("iteration")
plt.ylabel("acc")
plt.plot(epoch_list, acc_list, color='skyblue', label='test')
plt.plot(epoch_list, acc_list_test, color='green', label='train')
plt.show()

accuracy1 = check_accuracy(train_loader, model)
accuracy2 = check_accuracy(test_loader, model)
print("train accuracy: ", accuracy1)
print("test accuracy: ", accuracy2)

print("end")


# if __name__ == "__main__":
#     main()

