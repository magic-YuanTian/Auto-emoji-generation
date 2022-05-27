# Load Word2Vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from matplotlib import pyplot as plt


device = "cpu"

# wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/homes/cs577/hw2/w2v.bin"), binary=True)
wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/home/yuan/Desktop/w2v.bin"), binary=True)

output_path = "prediction.txt"

# Load training and testing data
def load_data(path, lowercase=True):
    sents = []
    tags = []
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            sent = []
            tag = []
            for pair in line.split('####')[1].split(' '):
                tn, tg = pair.rsplit('=', 1)
                if lowercase:
                    sent.append(tn.lower())
                else:
                    sent.append(tn)
                tag.append(tg)
            sents.append(sent)
            tags.append(tag)
    return sents, tags

def get_vocab_idx(train):
    tokens = set()
    for sent in train:
        tokens.update(sent)
    tokens = sorted(list(tokens))
    vocab2idx = dict(zip(tokens, range(1, len(tokens)+1)))
    vocab2idx["<PAD>"] = 0
    return vocab2idx

def convert_to_idx(sents, word2idx):
    for sent in sents:
        for i in range(len(sent)):
            if sent[i] in word2idx.keys():
                sent[i] = word2idx[sent[i]]
            else:
                sent[i] = 0

# number ---> one hot vector
def tag_to_embedding(tag_num, category_num):
    # -1 means there is no previous tag, therefore the embedding is [0, 0, 0, 0, 0]
    # if tag_num == -1:
    #     return [0] * category_num
    # else:
    #     emb = [0] * category_num
    #     emb[tag_num - 1] = 1  # one hot tag embedding
    #     return emb
    if tag_num >= category_num:
        emb = [0] * category_num # all 0 for the start tag
    else:
        emb = [0] * category_num
        emb[tag_num - 1] = 1  # one hot tag embedding
    return emb

#  tag2idx = {"O": 0, "T-NEG": 1, "T-NEU": 2, "T-POS": 3}
def idx2tag(idx, dict):
    for key in dict.keys():
        if dict[key] == idx:
            tag = key
            break

    return tag


# model for option 1
# 2-layer neural network
# embedding---FC----relu----FC----softmax
# hyperparameters: embedding dimension, context size
class model1(nn.Module):

    def __init__(self, vocab_size, embedding_dim, tag_size, hidden_size):
        super(model1, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(tag_size + embedding_dim, hidden_size) # P(y_i | y_{i-1}, x_i)
        self.linear2 = nn.Linear(hidden_size, tag_size)

    def forward(self, word_embed_idx, pre_tag_embed):
        embeds_vocab = self.embeddings(word_embed_idx).view((1, -1))
        cat_embed = torch.cat((embeds_vocab, pre_tag_embed), dim=1)
        z1 = self.linear1(cat_embed)
        a1 = F.relu(z1)
        z2 = self.linear2(a1)
        probabilities = F.log_softmax(z2, dim=1)
        return probabilities

# model for option 2
# 2-layer neural network
# embedding---FC----relu----FC----softmax
# hyperparameters: embedding dimension, context size
class model2(nn.Module):

    def __init__(self, embedding_dim, tag_size, hidden_size):
        super(model2, self).__init__()
        self.linear1 = nn.Linear(tag_size + embedding_dim, hidden_size) # P(y_i | y_{i-1}, x_i)
        self.linear2 = nn.Linear(hidden_size, tag_size)

    def forward(self, word_embedding, pre_tag_embed):
        cat_embed = torch.cat((word_embedding, pre_tag_embed), dim=1)
        z1 = self.linear1(cat_embed)
        a1 = F.relu(z1)
        z2 = self.linear2(a1)
        probabilities = F.log_softmax(z2, dim=1)
        return probabilities

# model for option 3
# contextalization: bidirectional lstm
class model3(nn.Module):

    def __init__(self, embedding_dim, tag_size, hidden_size_fc, hidden_size_lstm, num_layers_lstm):
        super(model3, self).__init__()
        self.num_layers_lstm = num_layers_lstm
        self.hidden_size_lstm = hidden_size_lstm

        self.lstm = nn.LSTM(embedding_dim, hidden_size_lstm, num_layers_lstm, bidirectional=True)
        self.linear1 = nn.Linear(tag_size + embedding_dim, hidden_size_fc) # P(y_i | y_{i-1}, x_i)
        self.linear2 = nn.Linear(hidden_size_lstm * 2 + hidden_size_fc, tag_size)

    def forward(self, word_embedding, pre_tag_embed, sentence_sequence, word_pos_idx):
        cat_embed = torch.cat((word_embedding, pre_tag_embed), dim=1)
        z1 = self.linear1(cat_embed)
        a1 = F.relu(z1)

        # bidirectional lstm
        h0 = torch.zeros(self.num_layers_lstm * 2, 1, self.hidden_size_lstm) # short term memory
        c0 = torch.zeros(self.num_layers_lstm * 2, 1, self.hidden_size_lstm) # long term memory
        out, _ = self.lstm(sentence_sequence, (h0, c0))

        out = out[word_pos_idx, :, :] # just get the output corresponding to the position
        a1_new = torch.cat((out, a1), dim=1) # contextualize a1

        z2 = self.linear2(a1_new)
        probabilities = F.log_softmax(z2, dim=1)
        return probabilities




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/twitter1_train.txt', help='Train file')
    parser.add_argument('--test_file', type=str, default='data/twitter1_test.txt', help='Test file')
    parser.add_argument('--prediction_file', type=str, default='prediction.txt', help='out put the prediction file')
    parser.add_argument('--option', type=int, default=3, help='Option to run (1 = Randomly Initialized, 2 = Word2Vec, 3 = contextulized(Bi-LSTM)')
    args = parser.parse_args()


    train_sents, train_tags = load_data(path=args.train_file)
    test_sents, test_tags = load_data(path=args.test_file)
    ori_test_sents, _ = load_data(path=args.test_file)

    # vocab2idx = get_vocab_idx(train_sents + test_sents)
    # convert_to_idx(train_sents, vocab2idx)
    # convert_to_idx(test_sents, vocab2idx)

    assert len(train_sents) == len(train_tags)  # judge if the train data is correct
    # We also need to convert the tags into integers that can then be fed to PyTorch's categorical cross entropy loss function

    # tag2idx = {"<PAD>": 0, "O": 1, "T-NEG": 2, "T-NEU": 3, "T-POS": 4, "<START>": 5}
    tag2idx = {"O": 0, "T-NEG": 1, "T-NEU": 2, "T-POS": 3}

    convert_to_idx(train_tags, tag2idx)
    convert_to_idx(test_tags, tag2idx)

    tag_num = len(tag2idx)

    # for ploting
    F1_list = []
    epoch_list = []
    # option branches
    if args.option == 1:
        print("option 1\n", flush=True)

        # build vacab dicts
        vocab2idx = get_vocab_idx(train_sents + test_sents)
        convert_to_idx(train_sents, vocab2idx)
        convert_to_idx(test_sents, vocab2idx)

        # some hyperparameters here
        epoch_num = 6
        embedding_dim = 100
        learning_rate = 0.005
        momentum = 0.4
        hidden_size = 128

        model = model1(len(vocab2idx), embedding_dim, tag_num, hidden_size)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # start training
        print("start training...")

        for epoch in range(epoch_num):
            print("---------epoch " + str(epoch) + "-------------", flush=True)
            for ins in range(len(train_sents)):
                # if ins > 50:
                #     break
                print("---------instance " + str(ins) + "-------------", flush=True)
                voc_list = train_sents[ins]
                tag_list = train_tags[ins]
                for i in range(len(voc_list)):
                    if i == 0:
                        pre_tag_embed = tag_to_embedding(5, tag_num)
                    else:
                        pre_tag_embed = tag_to_embedding(tag_list[i-1], tag_num)
                    pre_tag_embed = torch.FloatTensor(pre_tag_embed).view((1, -1))
                    word_idx = voc_list[i]
                    optimizer.zero_grad() # zero all gradients of managed parameters
                    inferred_prob = model(torch.tensor(word_idx), pre_tag_embed) # infer the log probability distribution on all possible tags
                    # target_idx = tag_to_embedding(tag_list[i], tag_num)
                    target_idx = torch.tensor(tag_list[i]).view(1)
                    loss = F.nll_loss(inferred_prob, target_idx) # calculate the loss
                    loss.backward() # after this, gradients are updated
                    optimizer.step() # use optimizer to update all managed parameters based on current gradients

            # test the performance of current model and store F1 score in the F1 list
            # try:
            #     _precision, _recall, F1 = viterbi(model, tag_num, test_sents, test_tags, args)
            #     F1_list.append(F1)
            #     epoch_list.append(epoch)
            # except Exception as e:
            #     continue

        print("finished training...")

    elif args.option == 2:
        print("option 2\n", flush=True)

        # construct dictionary
        vocab2idx = wv_from_bin.key_to_index
        convert_to_idx(train_sents, vocab2idx)
        convert_to_idx(test_sents, vocab2idx)

        # some hyperparameters here
        # embedding_dim = 300  # this is fixed

        epoch_num = 12
        learning_rate = 0.08
        momentum = 0.4
        hidden_size = 64

        model = model2(300, tag_num, hidden_size)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # start training
        print("start training...")

        for epoch in range(epoch_num):
            print("---------epoch " + str(epoch) + "-------------", flush=True)
            for ins in range(len(train_sents)):
                print("---------instance " + str(ins) + "-------------", flush=True)
                voc_list = train_sents[ins]
                tag_list = train_tags[ins]
                for i in range(len(voc_list)):
                    if i == 0:
                        pre_tag_embed = tag_to_embedding(5, tag_num)
                    else:
                        pre_tag_embed = tag_to_embedding(tag_list[i - 1], tag_num)
                    pre_tag_embed = torch.FloatTensor(pre_tag_embed).view((1, -1))
                    word_idx = voc_list[i]
                    optimizer.zero_grad()  # zero all gradients of managed parameters
                    # get the embedding from the pretrained model
                    word_emb = wv_from_bin.vectors[word_idx]
                    word_emb = torch.tensor(word_emb).view((1, -1))
                    # infer the log probability distribution on all possible tags
                    inferred_prob = model(word_emb, pre_tag_embed)
                    target_idx = torch.tensor(tag_list[i]).view(1)
                    loss = F.nll_loss(inferred_prob, target_idx)  # calculate the loss
                    loss.backward()  # after this, gradients are updated
                    optimizer.step()  # use optimizer to update all managed parameters based on current gradients

            # test the performance of current model and store F1 score in the F1 list
            # try:
            #     _precision, _recall, F1 = viterbi(model, tag_num, test_sents, test_tags, args)
            #     F1_list.append(F1)
            #     epoch_list.append(epoch)
            # except Exception as e:
            #     continue

        print("finished training...")

    elif args.option == 3:
        print("option 3\n", flush=True)

        # construct dictionary
        vocab2idx = wv_from_bin.key_to_index
        convert_to_idx(train_sents, vocab2idx)
        convert_to_idx(test_sents, vocab2idx)

        # some hyperparameters here
        # embedding_dim = 300  # this is fixed

        epoch_num = 5
        learning_rate = 0.08
        momentum = 0.4
        hidden_size = 64
        hidden_size_lstm = 64
        num_layers_lstm = 2

        model = model3(300, tag_num, hidden_size, hidden_size_lstm, num_layers_lstm)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # start training
        print("start training...")
        for epoch in range(epoch_num):
            print("---------epoch " + str(epoch) + "-------------", flush=True)
            for ins in range(len(train_sents)):
                print("---------instance " + str(ins) + "-------------", flush=True)
                voc_list = train_sents[ins]
                tag_list = train_tags[ins]

                # get embedding list for this sentence
                sentence_sequence = []
                for i in range(len(voc_list)):
                    word_idx = voc_list[i]
                    word_emb = wv_from_bin.vectors[word_idx]
                    word_emb = torch.tensor(word_emb).view((1, -1))
                    sentence_sequence.append(word_emb)
                # convert to tensor
                sentence_sequence = torch.stack(sentence_sequence, dim=0)
                # sentence_sequence = sentence_sequence.view((1, -1, 300))

                for i in range(len(voc_list)):
                    if i == 0:
                        pre_tag_embed = tag_to_embedding(5, tag_num)
                    else:
                        pre_tag_embed = tag_to_embedding(tag_list[i - 1], tag_num)
                    pre_tag_embed = torch.FloatTensor(pre_tag_embed).view((1, -1))
                    word_idx = voc_list[i]
                    optimizer.zero_grad()  # zero all gradients of managed parameters
                    # get the embedding from the pretrained model
                    word_emb = wv_from_bin.vectors[word_idx]
                    word_emb = torch.tensor(word_emb).view((1, -1))

                    # word_pos_idx = torch.tensor(i).view(1)
                    # infer the log probability distribution on all possible tags
                    inferred_prob = model(word_emb, pre_tag_embed, sentence_sequence, i)

                    target_idx = torch.tensor(tag_list[i]).view(1)
                    loss = F.nll_loss(inferred_prob, target_idx)  # calculate the loss
                    loss.backward()  # after this, gradients are updated
                    optimizer.step()  # use optimizer to update all managed parameters based on current gradients

            # test the performance of current model and store F1 score in the F1 list
            # try:
            #     _precision, _recall, F1 = viterbi(model, tag_num, test_sents, test_tags, args)
            #     F1_list.append(F1)
            #     epoch_list.append(epoch)
            # except Exception as e:
            #     continue

        print("finished training...")

    # viterbi algorithm and compute the precision, recall, F1
    precision, recall, F1 = viterbi(model, tag_num, test_sents, test_tags, args, ori_test_sents, tag2idx)

    # print(precision)
    # print(recall)
    # print(F1)

    # # plot
    # # question 1
    # plt.xlabel("epoch")
    # plt.ylabel("F1")
    # plt.plot(epoch_list, F1_list)
    # plt.show()

    print('end of program')


    # # ------------------------------- test part -------------------------------------------
    # # for finding the best combination of parameters
    # # test part
    # # question 1
    #
    # F1_list = []
    # lr_list = []
    # # learning_rate = 0.125
    # learning_rate = 0.161
    # lr_interval = 0.5
    #
    # # construct dictionary
    # vocab2idx = wv_from_bin.key_to_index
    # convert_to_idx(train_sents, vocab2idx)
    # convert_to_idx(test_sents, vocab2idx)
    #
    # while learning_rate < 0.162:
    #     # some hyperparameters here
    #     # embedding_dim = 300  # this is fixed
    #
    #     epoch_num = 6
    #     momentum = 0.1
    #     hidden_size = 128
    #
    #     model = model2(300, tag_num, hidden_size)
    #     optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    #     # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #
    #     # start training
    #     print("learning rate: ", learning_rate, flush=True)
    #     for epoch in range(epoch_num):
    #         # print("---------epoch " + str(epoch) + "-------------", flush=True)
    #         for ins in range(len(train_sents)):
    #             # print("---------instance " + str(ins) + "-------------", flush=True)
    #             voc_list = train_sents[ins]
    #             tag_list = train_tags[ins]
    #             for i in range(len(voc_list)):
    #                 if i == 0:
    #                     pre_tag_embed = tag_to_embedding(5, tag_num)
    #                 else:
    #                     pre_tag_embed = tag_to_embedding(tag_list[i - 1], tag_num)
    #                 pre_tag_embed = torch.FloatTensor(pre_tag_embed).view((1, -1))
    #                 word_idx = voc_list[i]
    #                 optimizer.zero_grad()  # zero all gradients of managed parameters
    #                 # get the embedding from the pretrained model
    #                 word_emb = wv_from_bin.vectors[word_idx]
    #                 word_emb = torch.tensor(word_emb).view((1, -1))
    #                 # infer the log probability distribution on all possible tags
    #                 inferred_prob = model(word_emb, pre_tag_embed)
    #                 target_idx = torch.tensor(tag_list[i]).view(1)
    #                 loss = F.nll_loss(inferred_prob, target_idx)  # calculate the loss
    #                 loss.backward()  # after this, gradients are updated
    #                 optimizer.step()  # use optimizer to update all managed parameters based on current gradients
    #
    #     try:
    #         precision, recall, F1 = viterbi(model, tag_num, test_sents, test_tags, args)
    #         F1_list.append(F1)
    #         lr_list.append(learning_rate)
    #         learning_rate += lr_interval
    #     except Exception as e:
    #         learning_rate += lr_interval
    #         continue
    #
    # # plot
    # # question 1
    # plt.xlabel("learning rate")
    # plt.ylabel("F1")
    # plt.plot(lr_list, F1_list)
    # plt.show()

    # ---------------------------- end test part ------------------------------------

    # deprecated
    # # open the file and prepare for writing predictions
    # save_file = open(output_path, 'w')
    #
    # # viterbi algorithm and compute the precision, recall, F1
    # with torch.no_grad():
    #     # precision, recall, F1
    #     total_precision = 0
    #     total_recall = 0
    #     total_F1 = 0
    #
    #     # viterbi algorithm
    #     # input is a sentence token index list
    #     # output the predicted tag sequence
    #     for index in range(len(test_sents)):
    #         sentence_list = test_sents[index]
    #
    #         '''
    #         maintain and update a DP structures
    #            {
    #                0 : {
    #                    "path": [] # sequence
    #                    "forward_props": [] # probability distribution from current node to next nodes
    #                    "cur_normalized_probs": float # current normalized probability
    #                },
    #                1: ...
    #            }
    #         '''
    #         DP = {}
    #         for i in range(tag_num):
    #             DP[i] = {}
    #
    #         for i in range(tag_num):
    #             DP[i]["path"] = []  # initialize the path (predicted tag list)
    #             DP[i]["forward_probs"] = [0] * tag_num
    #             DP[i]["cur_normalized_probs"] = 1  # evenly distributed (the number doesn't matter)
    #
    #         if args.option == 3:
    #             # get embedding list for this sentence
    #             sentence_sequence = []
    #             for i in range(len(sentence_list)):
    #                 word_idx = sentence_list[i]
    #                 word_emb = wv_from_bin.vectors[word_idx]
    #                 word_emb = torch.tensor(word_emb).view((1, -1))
    #                 sentence_sequence.append(word_emb)
    #             # convert to tensor
    #             sentence_sequence = torch.stack(sentence_sequence, dim=0)
    #             # sentence_sequence = sentence_sequence.view((1, -1, 300))
    #
    #         # handle the first token (start node --> first nodes)
    #         previous_tag = 5  # the start tag index (y_{i-1})
    #         pre_tag_embed = torch.FloatTensor(tag_to_embedding(5, tag_num)).view((1, -1))
    #         if args.option == 1:
    #             inferred_probs = model(torch.tensor(sentence_list[0]), pre_tag_embed)
    #         elif args.option == 2:
    #             word_emb = wv_from_bin.vectors[sentence_list[0]] # </s>
    #             word_emb = torch.tensor(word_emb).view((1, -1))
    #             inferred_probs = model(word_emb, pre_tag_embed)
    #         elif args.option == 3:
    #             word_emb = wv_from_bin.vectors[sentence_list[0]] # </s>
    #             word_emb = torch.tensor(word_emb).view((1, -1))
    #             inferred_probs = model(word_emb, pre_tag_embed, sentence_sequence, 0)
    #
    #         inferred_probs = torch.exp(inferred_probs)  # make log negative to normal prob
    #         inferred_probs = inferred_probs.tolist()[0]  # to list
    #         # update DP
    #         for i in range(tag_num):
    #             DP[i]["path"].append(i)  # update the path
    #             DP[i]["cur_normalized_probs"] = inferred_probs[i]
    #
    #
    #
    #         # start at the second token
    #         for i in range(1, len(sentence_list)):
    #             # try all possible choices from the previous step
    #             for key in DP.keys():
    #                 pre_tag_embed = torch.FloatTensor(tag_to_embedding(key, tag_num)).view((1, -1))
    #                 if args.option == 1:
    #                     inferred_probs = model(torch.tensor(sentence_list[i]), pre_tag_embed)
    #                 elif args.option == 2:
    #                     word_idx = sentence_list[i]
    #                     # get the embedding from the pretrained model
    #                     word_emb = wv_from_bin.vectors[word_idx]
    #                     word_emb = torch.tensor(word_emb).view((1, -1))
    #                     # infer the log probability distribution on all possible tags
    #                     inferred_probs = model(word_emb, pre_tag_embed)
    #                 elif args.option == 3:
    #                     word_idx = sentence_list[i]
    #                     # get the embedding from the pretrained model
    #                     word_emb = wv_from_bin.vectors[word_idx]
    #                     word_emb = torch.tensor(word_emb).view((1, -1))
    #                     # infer the log probability distribution on all possible tags
    #                     inferred_probs = model(word_emb, pre_tag_embed, sentence_sequence, i)
    #
    #                 inferred_probs = torch.exp(inferred_probs)  # make log negative to normal prob
    #                 inferred_probs = inferred_probs.tolist()[0]  # to list
    #                 # update forward probability in DP
    #                 for j in range(tag_num):
    #                     DP[key]["forward_probs"][j] = DP[key]["cur_normalized_probs"] * inferred_probs[j]
    #
    #             new_path = [[]] * tag_num
    #             # for each tag, find the path with highest probability
    #             # j ---> to tag index
    #             for j in range(tag_num):
    #                 # find prob of j from each previous node
    #                 max_prob = 0
    #                 best_pre = -1
    #
    #                 # k ---> from tag index
    #                 for k in range(tag_num):
    #                     if DP[k]["forward_probs"][j] > max_prob:
    #                         max_prob = DP[k]["forward_probs"][j]
    #                         best_pre = k
    #                 # record this path
    #                 try:
    #                     new_path[j] = DP[best_pre]["path"].copy()
    #                     new_path[j].append(j)
    #                 except Exception as e:
    #                     print("here")
    #                 # update current probability for this tag
    #                 DP[j]["cur_normalized_probs"] = max_prob
    #
    #             # update path and current probability for this tag
    #             for j in range(tag_num):
    #                 DP[j]["path"] = new_path[j]
    #
    #             # normalize current probability (divided by the maximum prob)
    #             temp_max = -1  # prob
    #             for j in range(tag_num):
    #                 if DP[j]["cur_normalized_probs"] > temp_max:
    #                     temp_max = DP[j]["cur_normalized_probs"]
    #             for j in range(tag_num):
    #                 if temp_max != 0:
    #                     DP[j]["cur_normalized_probs"] = DP[j]["cur_normalized_probs"] / temp_max
    #
    #         # finish constructing all the path
    #         best_idx = -1
    #         temp_max = -1  # prob
    #         for j in range(tag_num):
    #             if DP[j]["cur_normalized_probs"] > temp_max:
    #                 temp_max = DP[j]["cur_normalized_probs"]
    #                 best_idx = j
    #
    #         best_sequence = DP[j]["path"]
    #         ground_truth_sequence = test_tags[index]
    #
    #         # check if there length are the same
    #         assert len(best_sequence) == len(ground_truth_sequence)
    #
    #         print("\n\nGround truth:")
    #         print(test_tags[index])
    #         print("Prediction:")
    #         print(best_sequence)
    #
    #         # count TP, FP, FN
    #         TP = 0
    #         FP = 0
    #         FN = 0
    #         effective_tags = [2, 3, 4]  # T-POS, T-NEU, T-NEG
    #         for i in range(len(ground_truth_sequence)):
    #             if best_sequence[i] in effective_tags and best_sequence[i] == ground_truth_sequence[i]:
    #                 TP += 1
    #
    #             if best_sequence[i] != ground_truth_sequence[i]:
    #                 if best_sequence[i] in effective_tags:
    #                     FP += 1
    #                 if ground_truth_sequence[i] in effective_tags:
    #                     FN += 1
    #
    #         # Calculate precision, recall, F1 score
    #         if (TP + FP) == 0:
    #             precision = 0
    #         else:
    #             precision = TP / (TP + FP)
    #
    #         if (TP + FN) == 0:
    #             recall = 0
    #         else:
    #             recall = TP / (TP + FN)
    #
    #         if (precision + recall) == 0:
    #             F1 = 0
    #         else:
    #             F1 = precision * recall * 2 / (precision + recall)
    #
    #         # add them to total
    #         total_precision += precision
    #         total_recall += recall
    #         total_F1 += F1
    #         # store the predicted results into a file
    #
    #     # precision, recall, F1, calculate the mean
    #     total_precision /= len(test_sents)
    #     total_recall /= len(test_sents)
    #     total_F1 /= len(test_sents)
    #
    #     print("precision: " + str(total_precision))
    #     print("recall: " + str(total_recall))
    #     print("F1: " + str(total_F1))
    #
    # save_file.close()



def viterbi(model, tag_num, test_sents, test_tags, args, ori_test_sents, tag2idx):
    # open the file and prepare for writing predictions
    save_file = open(output_path, 'w')

    # viterbi algorithm and compute the precision, recall, F1
    with torch.no_grad():
        # precision, recall, F1
        total_precision = 0
        total_recall = 0
        total_F1 = 0

        # viterbi algorithm
        # input is a sentence token index list
        # output the predicted tag sequence
        for index in range(len(test_sents)):
            sentence_list = test_sents[index]

            # store in "prediction.txt"
            ori_str = ' '.join(ori_test_sents[index])
            save_file.write(ori_str)
            save_file.write('####')

            '''
            maintain and update a DP structures
               {
                   0 : {
                       "path": [] # sequence
                       "forward_props": [] # probability distribution from current node to next nodes
                       "cur_normalized_probs": float # current normalized probability
                   },
                   1: ...
               }
            '''
            DP = {}
            for i in range(tag_num):
                DP[i] = {}

            for i in range(tag_num):
                DP[i]["path"] = []  # initialize the path (predicted tag list)
                DP[i]["forward_probs"] = [0] * tag_num
                DP[i]["cur_normalized_probs"] = 1  # evenly distributed (the number doesn't matter)

            if args.option == 3:
                # get embedding list for this sentence
                sentence_sequence = []
                for i in range(len(sentence_list)):
                    word_idx = sentence_list[i]
                    word_emb = wv_from_bin.vectors[word_idx]
                    word_emb = torch.tensor(word_emb).view((1, -1))
                    sentence_sequence.append(word_emb)
                # convert to tensor
                sentence_sequence = torch.stack(sentence_sequence, dim=0)
                # sentence_sequence = sentence_sequence.view((1, -1, 300))

            # handle the first token (start node --> first nodes)
            previous_tag = 5  # the start tag index (y_{i-1})
            pre_tag_embed = torch.FloatTensor(tag_to_embedding(5, tag_num)).view((1, -1))
            if args.option == 1:
                inferred_probs = model(torch.tensor(sentence_list[0]), pre_tag_embed)
            elif args.option == 2:
                word_emb = wv_from_bin.vectors[sentence_list[0]]  # </s>
                word_emb = torch.tensor(word_emb).view((1, -1))
                inferred_probs = model(word_emb, pre_tag_embed)
            elif args.option == 3:
                word_emb = wv_from_bin.vectors[sentence_list[0]]  # </s>
                word_emb = torch.tensor(word_emb).view((1, -1))
                inferred_probs = model(word_emb, pre_tag_embed, sentence_sequence, 0)

            inferred_probs = torch.exp(inferred_probs)  # make log negative to normal prob
            inferred_probs = inferred_probs.tolist()[0]  # to list
            # update DP
            for i in range(tag_num):
                DP[i]["path"].append(i)  # update the path
                DP[i]["cur_normalized_probs"] = inferred_probs[i]

            # start at the second token
            for i in range(1, len(sentence_list)):
                # try all possible choices from the previous step
                for key in DP.keys():
                    pre_tag_embed = torch.FloatTensor(tag_to_embedding(key, tag_num)).view((1, -1))
                    if args.option == 1:
                        inferred_probs = model(torch.tensor(sentence_list[i]), pre_tag_embed)
                    elif args.option == 2:
                        word_idx = sentence_list[i]
                        # get the embedding from the pretrained model
                        word_emb = wv_from_bin.vectors[word_idx]
                        word_emb = torch.tensor(word_emb).view((1, -1))
                        # infer the log probability distribution on all possible tags
                        inferred_probs = model(word_emb, pre_tag_embed)
                    elif args.option == 3:
                        word_idx = sentence_list[i]
                        # get the embedding from the pretrained model
                        word_emb = wv_from_bin.vectors[word_idx]
                        word_emb = torch.tensor(word_emb).view((1, -1))
                        # infer the log probability distribution on all possible tags
                        inferred_probs = model(word_emb, pre_tag_embed, sentence_sequence, i)

                    inferred_probs = torch.exp(inferred_probs)  # make log negative to normal prob
                    inferred_probs = inferred_probs.tolist()[0]  # to list
                    # update forward probability in DP
                    for j in range(tag_num):
                        DP[key]["forward_probs"][j] = DP[key]["cur_normalized_probs"] * inferred_probs[j]

                new_path = [[]] * tag_num
                # for each tag, find the path with highest probability
                # j ---> to tag index
                for j in range(tag_num):
                    # find prob of j from each previous node
                    max_prob = 0
                    best_pre = -1

                    # k ---> from tag index
                    for k in range(tag_num):
                        if DP[k]["forward_probs"][j] > max_prob:
                            max_prob = DP[k]["forward_probs"][j]
                            best_pre = k
                    # record this path
                    try:
                        new_path[j] = DP[best_pre]["path"].copy()
                        new_path[j].append(j)
                    except Exception as e:
                        # print("here")
                        pass
                    # update current probability for this tag
                    DP[j]["cur_normalized_probs"] = max_prob

                # update path and current probability for this tag
                for j in range(tag_num):
                    DP[j]["path"] = new_path[j]

                # normalize current probability (divided by the maximum prob)
                temp_max = -1  # prob
                for j in range(tag_num):
                    if DP[j]["cur_normalized_probs"] > temp_max:
                        temp_max = DP[j]["cur_normalized_probs"]
                for j in range(tag_num):
                    if temp_max != 0:
                        DP[j]["cur_normalized_probs"] = DP[j]["cur_normalized_probs"] / temp_max

            # finish constructing all the path
            best_idx = -1
            temp_max = -1  # prob
            for j in range(tag_num):
                if DP[j]["cur_normalized_probs"] > temp_max:
                    temp_max = DP[j]["cur_normalized_probs"]
                    best_idx = j

            best_sequence = DP[best_idx]["path"]
            ground_truth_sequence = test_tags[index]

            # write the tag part into prediction.txt
            for p in range(len(sentence_list)):
                save_file.write(ori_test_sents[index][p])
                save_file.write('=')
                temp_tag_idx = best_sequence[p]
                temp_tag = idx2tag(temp_tag_idx, tag2idx)
                save_file.write(temp_tag) # tag
                if p != len(sentence_list) - 1:
                    save_file.write(' ')  # tag

            save_file.write('\n')

            # check if there length are the same
            assert len(best_sequence) == len(ground_truth_sequence)

            # print("\n\nGround truth:")
            # print(test_tags[index])
            # print("Prediction:")
            # print(best_sequence)

            # count TP, FP, FN
            TP = 0
            FP = 0
            FN = 0
            effective_tags = [2, 3, 4]  # T-POS, T-NEU, T-NEG
            for i in range(len(ground_truth_sequence)):
                if best_sequence[i] in effective_tags and best_sequence[i] == ground_truth_sequence[i]:
                    TP += 1

                if best_sequence[i] != ground_truth_sequence[i]:
                    if best_sequence[i] in effective_tags:
                        FP += 1
                    if ground_truth_sequence[i] in effective_tags:
                        FN += 1

            # Calculate precision, recall, F1 score
            if (TP + FP) == 0:
                precision = 0
            else:
                precision = TP / (TP + FP)

            if (TP + FN) == 0:
                recall = 0
            else:
                recall = TP / (TP + FN)

            if (precision + recall) == 0:
                F1 = 0
            else:
                F1 = precision * recall * 2 / (precision + recall)

            # add them to total
            total_precision += precision
            total_recall += recall
            total_F1 += F1
            # store the predicted results into a file

        # precision, recall, F1, calculate the mean
        total_precision /= len(test_sents)
        total_recall /= len(test_sents)
        total_F1 /= len(test_sents)

        print("precision: " + str(total_precision))
        print("recall: " + str(total_recall))
        print("F1: " + str(total_F1))

    save_file.close()
    return (total_precision, total_recall, total_F1)



if __name__ == "__main__":
    main()




