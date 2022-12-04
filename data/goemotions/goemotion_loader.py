import pandas as pd
import os
import json

train_file = 'train.tsv'
dev_file = 'dev.tsv'
test_file = 'test.tsv'

emotion_file = 'emotions.txt'
ekman_emotion_mapping = 'ekman_mapping.json'
sentiment_mapping = 'sentiment_mapping.json'

# def get_emo(grouping=None):
#     emo_list = EMOTION_LIST
#     emo_list = [line.strip() for idx, line in enumerate(open(emotion_file, 'r').readlines())]
#     return emo_list
EMOTION_LIST = [line.strip() for idx, line in enumerate(open(os.path.join('data/goemotions/emotions.txt'), 'r').readlines())]


def get_emo_grouped(grouping, file_path=''):
    if grouping == "ekman":
        fp = os.path.join(file_path, ekman_emotion_mapping)
    if grouping == "sentiment":
        fp = os.path.join(file_path, sentiment_mapping)
    with open(fp) as file:
        mapping = json.load(file)
        emo_list = list(mapping.keys())
    return emo_list, mapping

def emo2group(mapping):
    idx_mapping = {}
    for i, emotion in enumerate(EMOTION_LIST):
        for j, ele in enumerate(mapping):
            if emotion in mapping[ele]:
                idx_mapping[i] = j
    return idx_mapping

def group_mapping(emotion_list, mapping):
    map_list = []
    for emotion in emotion_list:
        for ele in mapping:
            if emotion in mapping[ele]:
                map_list.append(ele)
            if emotion == 'neutral':
                map_list.append('neutral')

    return map_list

def get_emotion(file, emo_list, grouping=None, mapping=None):
    text_list = []
    label_list = []
    if mapping is not None:
        idx_mapping = emo2group(mapping)
    with open(file, 'r') as f:
        for line in f.readlines():
            text, emotions, _ = line.split('\t')
            text_list.append(text)
            one_label = [0] * len(emo_list)
            for emo in emotions.split(','): # ['joy', 'admiration', 'caring']
                if mapping is not None:
                    emo_idx = idx_mapping[int(emo)]
                else:
                    emo_idx = int(emo)
                one_label[emo_idx] = 1
                label_list.append(one_label)
    # import pdb; pdb.set_trace()
    return text_list, label_list


def goemotion_data(file_path='', remove_stop_words=True, get_text=True, grouping=None):

    if grouping == None: 
        emo_list = EMOTION_LIST
        mapping = None
    else:
        emo_list, mapping = get_emo_grouped(grouping, file_path)

    X_train, y_train = get_emotion(os.path.join(file_path, train_file), emo_list, mapping)
    X_dev, y_dev = get_emotion(os.path.join(file_path, dev_file), emo_list, mapping)
    X_test, y_test = get_emotion(os.path.join(file_path, test_file), emo_list, mapping)

    return X_train, y_train, X_dev, y_dev, X_test, y_test, emo_list
