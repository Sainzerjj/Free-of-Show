import os
import pickle as pkl
import argparse
import torch
from fairseq.models.roberta import alignment_utils
import spacy
import inflect
import json

nlp = spacy.load("en_core_web_sm")
engine = inflect.engine()
word_set = {}
# 'pytorch/fairseq'
model_path = 'pytorch/fairseq'
# model_path = '/home/amax/.cache/torch/hub/pytorch_fairseq_main'
roberta = torch.hub.load(model_path, 'roberta.base')
# curr_path = os.getcwd()
# with open(os.path.join(curr_path, 'datasets/category_dict.txt'), 'r') as f:
#     all_categories = f.read()
#     words_list = all_categories.split(', ')
# words_list = [word.strip() for word in words_list]
# for each in words_list:
# 	word_set[each] = [each.lower()]
curr_path = os.getcwd()
word_set = torch.load(os.path.join(curr_path, 'datasets/word_set.pkl'))

def check_in_dataset(noun_pharse):
    # If noun_phrase is in ms_coco, return True
    for each_cate in word_set:
        if each_cate in noun_pharse:
            return True
    return False


nlp = spacy.load("en_core_web_sm")

def inference_sentence(sentence, opt=None):
    with open(os.path.join(curr_path, opt.dir_path), "r") as f:
        contents = f.read()
        rows = contents.split('\n')
        rows = rows[:2000]
        sentence_list = []
        for i in range(500):
            sentence_list.append(rows[4*i + 2][10:])
    results = []
    with torch.no_grad():
        for sentence in sentence_list:
            if sentence is None:
                sentence = opt.sentence if (opt is not None) else 'The silver bed was situated to the right of the white couch.'
            sentence = sentence.replace("\n", "")
            sentence = sentence.rstrip()
            sentence = sentence.lstrip()
            doc = nlp(sentence)
            pos = []
            phrases = []
            entities = []
            for chunk in doc.noun_chunks:
                full_noun = (chunk.text).lower()
                key_noun = (chunk.root.text).lower()
                count = len(full_noun.split())
                word_index = chunk.root.i
                index_list = [word_index - i for i in range(count)]
                index_list.reverse()
                
                if check_in_dataset(key_noun):
                    phrases.append(key_noun)  # phrases = ['fork', 'table', 'bottle']
                    entities.append(index_list) # entities = [2, 9, 14]
                    pos.append(full_noun)   # pos = ['the brown forks', 'the table', 'the red bottle']

            print(f"Sentence: {sentence}")
            result = {"prompt": sentence, "phrases": phrases, "entities": entities, "original_boxes": None}
            results.append(result)
        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentence",
        type=str,
        nargs="?",
        default="The brown fork was placed on top of the table, with the red bottle resting securely below it."
    )
    parser.add_argument(
        "--dir_path",
        type=str,
        default='datasets/gpt.txt'
    )
    opt = parser.parse_args()
    results = inference_sentence(None, opt)
    with open('prompt_dataset.json', 'w') as f:
        for i, result in enumerate(results):
            json.dump(result, f)
            f.write(', \n')
            print(i)
    # inference_sentence(opt)
    # sentence = 'The silver bed was situated to the right of the white couch.'
    # sentence = 'The brown fork was placed on top of the table, with the red bottle resting securely below it.'
    # sentence = 'The silver laptop was perched atop the green keyboard, its screen illuminated with a bright glow.'
    # sentence = "The apple is placed above the banana."
    # sentence = "The apple is placed right of the banana."
    # sentence = "The apple is placed right of the banana, and a sandwitch is placed to their left."