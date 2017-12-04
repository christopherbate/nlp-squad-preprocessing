import VocabBuilder
import EMBuilder
import ExperimentSetup
import spacy
import keras
import numpy as np
from keras.layers import Input, Dense, LSTM
from keras.models import Model

if __name__ == '__main__':
    train = False
    nlp = spacy.load('en')
    data_directory = 'data'
    exp_directory = 'exp'

    print("Loading vocab.")
    vocabBuilder = VocabBuilder.VocabBuilder(['data/train.context',
                                    'data/train.question'],'data/vocab.dat')
    vocabBuilder.loadVocab()

    print("Loading embedding matrix.")
    em = EMBuilder.EmbeddingMatrix(vocabBuilder.word_idx_lookup, \
        gloveSize='100',gloveDir='data', vocabFile='data/vocab.dat', \
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
    truth = np.load('exp/train.idx.maskArray.npy')

    # First Keras model - a simple densely connected network to predict a one-word answer
    context_input = Input(shape=(300,), name='context_input')
    question_input = Input(shape=(20,), name='question_input')

    # Embedding layers for context and question
    embedding_layer = keras.layers.Embedding( len(vocabBuilder.word_list), 100,weights=[em.embedding_matrix],input_length=300,trainable=False)
    embedding_layer_question = keras.layers.Embedding(len(vocabBuilder.word_list),100,weights=[em.embedding_matrix],input_length=20,trainable=False)
    embedded_sequences_context = embedding_layer(context_input)
    embedded_sequences_question = embedding_layer_question(question_input)

    # First, read the question
    encoder_outputs, state_h, state_c = LSTM(64, return_state=True)(embedded_sequences_question)

    # Now, read the paragraph, conditioned on question output.
    encoder_states = [state_h, state_c]
    context_encoder = LSTM(64)(embedded_sequences_context, initial_state=encoder_states)

    #finalOut = Dense(50,activation='relu')(x)
    intermediate = Dense(300, activation='relu')(context_encoder)
    predictions = Dense(300,activation='sigmoid', name='main_output')(intermediate)
    model = Model(inputs=[context_input,question_input],outputs=predictions)
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    if(train):
        print("Training model.")
        model.fit( {'context_input':context,'question_input':question},{'main_output':truth},epochs=5,batch_size=32 )
        model.save_weights('exp/weights.h5')



    model.load_weights('exp/weights.h5')
    xValContext = np.load('exp/xval.idx.contextArray.npy')
    xValQuestion = np.load('exp/xval.idx.questionArray.npy')
    xValResults = model.predict({'context_input':xValContext,'question_input':xValQuestion},batch_size=200,verbose=1)
    predTranslated = []
    questTranslated = []
    contTranslated = []
    for i in range(xValResults.shape[0]):
        words = []
        for count, num in enumerate(xValContext[i,:].tolist()):
            words.append(vocabBuilder.word_list[num])
        contTranslated.append(words)
        words = []
        for count, num in enumerate(xValResults[i,:].tolist()):
            if(num>0.5):
                words.append(contTranslated[-1][count])
        predTranslated.append(words)
        words = []
        for count, num in enumerate(xValQuestion[i,:].tolist()):
            if(num==1):
                words.append(vocabBuilder.word_list[num])
        questTranslated.append(words)

    with open("exp/xvalResults.txt","w",encoding='utf-8') as xvalResults:
        for count, line in enumerate(predTranslated):
            contextLine = contTranslated[count]
            xvalResults.write('Context: '+' '.join(word for word in contextLine)+'\n')
            questionLine = questTranslated[count]
            xvalResults.write('Question: '+' '.join(word for word in questionLine)+'\n')
            xvalResults.write('Answer: '+' '.join(word for word in line)+'\n')
