import VocabBuilder
import EMBuilder
import ExperimentSetup
import spacy
import keras
import numpy as np
from keras.layers import Input, Dense, LSTM
from keras.models import Model

if __name__ == '__main__':
    nlp = spacy.load('en')
    data_directory = 'data'
    exp_directory = 'exp'

    print("Loading vocab.")
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
    with open('exp/train.idx.answer', 'r',encoding='utf-8') as span_file:
        for line in span_file:
            spanArr.append(line.strip().split(' ')[0]);
    y_train = np.array(spanArr)
    #y_train = keras.utils.to_categorical(spanArr, num_classes=em.embedding_matrix.shape[0])


    # First Keras model - a simple densely connected network to predict a one-word answer
    context_input = Input(shape=(300,), name='context_input')
    question_input = Input(shape=(20,), name='question_input')
    embedding_layer = keras.layers.Embedding( len(vocabBuilder.word_list), 100,weights=[em.embedding_matrix],input_length=300,trainable=False)
    embedding_layer_question = keras.layers.Embedding(len(vocabBuilder.word_list),100,weights=[em.embedding_matrix],input_length=20,trainable=False)
    embedded_sequences_context = embedding_layer(context_input)
    embedded_sequences_question = embedding_layer_question(question_input)

    # First, read the question
    encoder_outputs, state_h, state_c = LSTM(32)(embedded_sequences_question,return_state=True)
    # Now, read the paragraph, conditioned on question output.
    encoder_states = [state_h, state_c]
    context_encoder = LSTM(32, return_sequences=True, return_state=True)()
    #x = keras.layers.Conv1D(128,5,padding="same",activation='relu')(embedded_sequences_context)

    #x = keras.layers.Flatten()(x)
    #x1 = keras.layers.Conv1D(128,5,padding='same',activation='relu')(embedded_sequences_question)

    #x1 = keras.layers.Flatten()(embedded_sequences_context)
    combined = keras.layers.concatenate([x,x1])
    #finalOut = Dense(50,activation='relu')(x)
    predictions = Dense(len(vocabBuilder.word_list),activation='softmax',name='main_output')(combined)

    print("Training model.")
    model = Model(inputs=[context_input,question_input],outputs=predictions)
    model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit( {'context_input':context,'question_input':question},{'main_output':y_train},epochs=5,batch_size=32 )

    model.save('exp/weights')
