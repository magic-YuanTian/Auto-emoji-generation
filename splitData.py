
import pandas as pd
import csv

df = pd.read_csv("data/emoji_cleaned.csv")
data = df.values.tolist()

num_of_data = len(data)



validation_portion = int(num_of_data * 0.8)

train_data = data[:validation_portion]
test_data = data[validation_portion + 1:]

with open('data/train_emoji_cleaned.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(train_data)

with open('data/test_emoji_cleaned.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(test_data)

print("end")