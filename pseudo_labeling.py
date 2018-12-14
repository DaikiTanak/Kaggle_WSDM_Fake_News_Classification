import pickle
import pandas as pd
import numpy as np
from preprocess import english_clean_series, chinese_clean_series

def pseudo_label_test(threshold=0.99):
    with open('save/average_prediction.pickle', mode='rb') as f:
        predictions = pickle.load(f)

    test_df = pd.read_csv("data/test.csv")
    id_ = test_df["id"]
    test_df["title1_zh"] =  chinese_clean_series(test_df["title1_zh"])
    test_df["title2_zh"] =  chinese_clean_series(test_df["title2_zh"])
    test_df["title1_en"] =  english_clean_series(test_df["title1_en"])
    test_df["title2_en"] =  english_clean_series(test_df["title2_en"])


    add = 0

    new_data = []
    for each_id, p, zh1, zh2, en1, en2 in zip(id_, predictions,
                                               test_df["title1_zh"],
                                               test_df["title2_zh"],
                                               test_df["title1_en"],
                                               test_df["title2_en"]):

        max_label = np.argmax(p)
        confidence = np.max(p)

        if confidence > threshold:
            new_data.append((en1, en2, zh1, zh2, max_label))
            add += 1

    print("all:{}, add:{}".format(len(predictions), add))
    return new_data

if __name__ == "__main__":
    pseudo_label_test()
