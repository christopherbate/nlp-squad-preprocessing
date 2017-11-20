import tqdm
import os

class VocabBuilder:
    def __init__(self, vocabInFiles=['data/train.context','data/train.question'],
                    vocabOutFilename='data/vocab.dat'):
        self.vocab = {}
        self.vocabOutFilename = vocabOutFilename
        self.vocabInFiles = vocabInFiles
        self.word_list = []
        self.word_idx_lookup = {}

    def createVocab( self ):
        for fname in self.vocabInFiles:
            with open(fname,'r',encoding='utf-8') as vIn:
                for line in tqdm.tqdm(vIn):
                    words = line.strip().split()
                    for word in words:
                        if not (word in self.vocab):
                            self.vocab[word] = 1
                        else:
                            self.vocab[word] += 1
        self.word_list = ['_UNK_'] + sorted(self.vocab,key=self.vocab.get,reverse=True)
        self.word_idx_lookup = dict([(x,y) for (y,x) in enumerate(self.word_list)])

    def saveVocab(self):
        if not ( len(self.word_list) > 0):
            return
        with open(self.vocabOutFilename, "w",encoding="utf-8") as vf:
            for word in tqdm.tqdm(self.word_list):
                vf.write( word + '\n')

    def loadVocab( self ):
        self.word_list = []
        self.word_idx_lookup = {}
        with open( self.vocabOutFilename,"r",encoding='utf-8') as vf:
            self.word_list.extend(vf.readlines())
            self.word_list = [word.strip('\n') for word in self.word_list]
            self.word_idx_lookup = dict([(x,y) for (y,x) in enumerate(self.word_list)])
