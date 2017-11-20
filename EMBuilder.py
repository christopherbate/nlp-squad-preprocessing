import numpy as np
import os
import tqdm

class EmbeddingMatrix:
    def __init__(self, word_idx_lookup, gloveSize='100',gloveDir='data',
               vocabFile='data/vocab.dat',saveLoc='data/embedding_matrix'):
        self.gloveSize = gloveSize
        self.gloveFile = 'glove.6B.'+self.gloveSize+'d.txt'
        self.data_dir = gloveDir
        self.saveLoc = saveLoc
        self.num_matched = 0
        self.embeddings_dict = {}
        self.word_idx_lookup = word_idx_lookup
        self.embedding_matrix = np.zeros( ( len(self.word_idx_lookup), int(self.gloveSize) ) )
        self.loadGloveDict()
        #self.buildEmbeddingMatrix()

    def loadGloveDict(self):
        with open(os.path.join(self.data_dir, self.gloveFile),encoding='utf-8') as gf:
            for line in gf:
                values = line.split(' ')
                word = values[0]
                coefs = np.asarray(values[1:],dtype='float32')
                self.embeddings_dict[word] = coefs

    def buildEmbeddingMatrix(self):
        for word in tqdm.tqdm(self.word_idx_lookup):
            if(word.lower() in self.embeddings_dict):
                self.embedding_matrix[self.word_idx_lookup[word],:] = \
                        self.embeddings_dict[word.lower()]
                self.num_matched += 1
        print("Num matched: ", self.num_matched)
        np.savez_compressed(self.saveLoc+self.gloveSize,matrix=self.embedding_matrix)

    def loadEmbeddingMatrix(self):
        self.embedding_marix = np.load(self.saveLoc+self.gloveSize+'.npz')['matrix']
