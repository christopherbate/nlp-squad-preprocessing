#------------------------------------------------------------------------------
# SQuAD Pre-processing class
# Adapted from CS224N ass4 code.
#------------------------------------------------------------------------------
import os

import tqdm
import numpy as np
import json
import matplotlib as mpl
mpl.use('TkAgg')

class SQUADProcesser():
    def __init__(self, tokenizer, data_dir='data'):
        self.data_dir = data_dir
        self.glove_dir = 'data'
        #self.data = None
        self.trainFile = 'train-v1.1.json'
        self.devFile = 'dev-v1.1.json'
        self.outPrefix = "train"
        self.tokenizer = tokenizer
        self.numTrainExamples = 0
        self.contextLengths = None
        self.vocab = {}

    def break_file(self, prefix, filename='train-v1.1.json', countExamples=False):
        self.load_data(filename)
        self.outPrefix = prefix
        with open(os.path.join(self.data_dir, self.outPrefix +'.context'), 'w', encoding='utf-8') as context_file, \
             open(os.path.join(self.data_dir, self.outPrefix +'.question'), 'w', encoding='utf-8') as question_file, \
             open(os.path.join(self.data_dir, self.outPrefix +'.answer'), 'w', encoding='utf-8') as text_file, \
             open(os.path.join(self.data_dir, self.outPrefix +'.span'), 'w', encoding='utf-8') as span_file, \
             open(os.path.join(self.data_dir, self.outPrefix + ".stats"), "w", encoding='utf-8') as stats_file, \
             open(os.path.join(self.data_dir, self.outPrefix + ".mask"), "w", encoding="utf-8") as mask_file:
             # for each article...
            for aid in tqdm.tqdm(range(len(self.data['data']))):
                paragraphs = self.data['data'][aid]['paragraphs']
                 # for each paragraph...
                for pid in range(len(paragraphs)):
                    context = paragraphs[pid]['context']
                    context = context.strip().replace("\n", " ")
                    context_tokens = self.tokenizer(context)
                    # for each question/ans set...
                    qas = paragraphs[pid]['qas']
                    for qid in range(len(qas)):
                        question = qas[qid]['question']
                        question_tokens = self.tokenizer(question)
                        # select the to top answer:
                        ans_id = 0
                        answer_text = qas[qid]['answers'][ans_id]['text']
                        answer_tokens = self.tokenizer(answer_text)
                        answer_start = qas[qid]['answers'][ans_id]['answer_start']
                        # find the token hat begins with that character.
                        for token in context_tokens:
                            if token.idx == answer_start:
                                answer_start = token.i
                                break

                        # Generate the answer "word mask"
                        mask = []
                        for token in context_tokens:
                            if token.i < answer_start:
                                mask.append('0')
                            elif token.i < (answer_start+len(answer_tokens)):
                                mask.append('1')
                            else:
                                mask.append('0')

                        # write to file.
                        context_file.write(' '.join(token.string for token in context_tokens)+'\n')
                        question_file.write(' '.join(token.string for token \
                            in question_tokens)+'\n')
                        text_file.write(' '.join(token.string for token in answer_tokens)+'\n')
                        span_file.write(str(answer_start) + '\n')
                        mask_file.write(' '.join(item for item in mask) +'\n')

                        # stats string: num of context tokens, num answer tokens
                        stats_file.write(str(len(context_tokens)) + ',' + \
                            str(len(question_tokens))+'\n')
                        if countExamples:
                            self.numTrainExamples += 1

    def load_data(self, filename='train-v1.1.json'):
        full_filepath = os.path.join(self.data_dir, filename)
        with open(full_filepath) as datafile:
            self.data = json.load(datafile)

    def conduct_preprocess(self):
        self.break_file("train", self.trainFile, True)
        self.break_file("dev", self.devFile, False)
