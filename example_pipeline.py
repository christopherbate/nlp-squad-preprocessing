import squad_processer
import VocabBuilder
import EMBuilder
import ExperimentSetup
import spacy
import numpy as np

if __name__ == '__main__':
    nlp = spacy.load('en')
    data_directory = 'data'
    exp_directory = 'exp'
    #sp = squad_processer.SQUADProcesser(nlp.tokenizer,data_dir=data_directory)

    print("Splitting files.")
    #sp.conduct_preprocess()

    print("Building vocab.")
    vocabBuilder = VocabBuilder.VocabBuilder(['data/train.context',
                                    'data/train.question'],'data/vocab.dat')
    #vocabBuilder.createVocab()
    #vocabBuilder.saveVocab()
    vocabBuilder.loadVocab()

    print("Building embedding matrices.")
    em = EMBuilder.EmbeddingMatrix(vocabBuilder.word_idx_lookup,
                gloveSize='100',gloveDir='data', vocabFile='data/vocab.dat',
                saveLoc = 'data/embedding_matrix')
    em.buildEmbeddingMatrix()

    #context = np.load('exp/train.idx.contextArray.npy')
    #question = np.load('exp/train.idx.questionArray.npy')

    print("Matched " , em.num_matched, " words to vectors out of vocab of ",
           len(vocabBuilder.vocab))
    print( "Creating experiment.")
    es = ExperimentSetup.ExperimentSetup(vocabBuilder.word_idx_lookup,
    wordVecSize='100',maxContextLength=300,maxQuestionLength=20,padZero=True,
    experimentFolder=exp_directory,dataFolder=data_directory,train_percent=0.8,
    shuffleExamples=True,sourceTier="train")

    es.generateExperimentFiles(es.trainIndices,"train.idx","train")
    es.generateExperimentFiles(es.xvalIndices,"xval.idx","train")
