import VocabBuilder
import EMBuilder
import ExperimentSetup
import spacy
import keras
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

if __name__ == '__main__':
    nlp = spacy.load('en')
    data_directory = 'data'
    exp_directory = 'exp'

    print("Building vocab.")
    vocabBuilder = VocabBuilder.VocabBuilder(['data/train.context',
                                    'data/train.question'],'data/vocab.dat')
    vocabBuilder.loadVocab()

    print("Loading embedding matrix.")
    em = EMBuilder.EmbeddingMatrix(vocabBuilder.word_idx_lookup,
                gloveSize='100',gloveDir='data', vocabFile='data/vocab.dat',
                saveLoc = 'data/embedding_matrix')
    em.loadEmbeddingMatrix()
    #print("Matched " , em.num_matched, " words to vectors out of vocab of ",
    #       len(vocabBuilder.vocab))
    print( "Creating experiment.")
    '''es = ExperimentSetup.ExperimentSetup(vocabBuilder.word_idx_lookup,
    wordVecSize='100',maxContextLength=300,maxQuestionLength=20,padZero=True,
    experimentFolder=exp_directory,dataFolder=data_directory,train_percent=0.8,
    shuffleExamples=True,sourceTier="train")
    es.generateExperimentFiles(es.trainIndices,"train.idx","train")
    es.generateExperimentFiles(es.xvalIndices,"xval.idx","train")'''

    context = np.load('exp/train.idx.contextArray.npy')
    question = np.load('exp/train.idx.questionArray.npy')
    spanArr = []
    with open('exp/train.idx.span', 'r',encoding='utf-8') as span_file:
        for line in span_file:
            spanArr.append(line);
    spanArr = np.array(spanArr)
    y_train = keras.utils.to_categorical(spanArr, num_classes=em.embedding_matrix.shape[0])


    # First Keras model - a simple densely connected network to predict a one-word answer
    context_input = Input(shape=(300,), name='context_input')
    question_input = Input(shape=(20,), name='question_input')
    x = Dense(100,activation='relu')(context_input)
    x1 = Dense(100,activation='relu')(question_input)
    combined = keras.layers.concatenate([x,x1])
    finalOut = Dense(100,activation='relu')(combined)
    predictions = Dense(len(vocabBuilder.word_list),activation='softmax',name='main_output')(combined)

    print("Training model.")
    model = Model(inputs=[context_input,question_input],outputs=predictions)
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit( {'context_input':context,'question_input':question},{'main_output':y_train},epochs=1,batch_size=32 )

    model.save('exp/weights')
