from MyModels import *

wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/home/yuan/Desktop/w2v.bin"), binary=True)

df = pd.read_csv("data/emoji_cleaned.csv")
data = df.values.tolist()

vocab2idx = wv_from_bin.key_to_index