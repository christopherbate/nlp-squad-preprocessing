import numpy as np
import os
import linecache
import tqdm
import pathlib

class ExperimentSetup:
    def __init__(self, wordIdxLookup, wordVecSize='100', maxContextLength=300,
                 maxQuestionLength=10, padZero=True,experimentFolder='exp',dataFolder='data',
                 train_percent=0.9,shuffleExamples=True, sourceTier="train"):
        print("Generating experiment files.")
        self.wordVecSize = wordVecSize
        self.maxContextLength = maxContextLength
        self.maxQuestionLength = maxQuestionLength
        self.dataFolder = dataFolder
        self.expFolder = experimentFolder
        self.train_percent = train_percent
        self.wordIdxLookup = wordIdxLookup
        self.npzLoc = os.path.join(dataFolder,'embedding_matrix'+wordVecSize+'.npz')
        self.embedding_matrix = np.load( self.npzLoc )['matrix']
        self.numTrainingExamples = self.countTrainingExamples(os.path.join(dataFolder,sourceTier+'.context'))
        self.numUnkWordEncountered = 0
        self.padNumber = 0

        # create the experiment folder
        pathlib.Path('./'+experimentFolder).mkdir(parents=True,exist_ok=True)

        # some debug information
        print("Embedding matrix shape: ", self.embedding_matrix.shape)
        print("Number of training examples available: ", self.numTrainingExamples)
        # Select the indices into the training data that have context lengths less than max length
        indices = self.getTrainExIndices(sourceTier)
        self.numTrainingExamples = len(indices)
        print("Number of qualifying training examples: ", self.numTrainingExamples)
        if(shuffleExamples== True):
            np.random.shuffle(indices)
            print("Shuffled indices. The first index is now: ", indices[0])
        print("Splitting indices into train and xval groups.")
        splitPoint = int( self.numTrainingExamples * train_percent)
        self.finalTrainExCount = splitPoint
        self.trainIndices = indices[:splitPoint]
        self.xvalIndices = indices[splitPoint:]
        print(len(self.trainIndices)," training examples and ", len(self.xvalIndices), " x-val examples.")

    def generateExperimentFiles(self, indices, prefix, sourceTier):
        contextArray = []
        questionArray = []
        linecache.clearcache()
        with open(os.path.join(self.expFolder, prefix + '.context'), 'w',encoding="utf-8") as context_file,  \
             open(os.path.join(self.expFolder, prefix + '.question'), 'w',encoding="utf-8") as question_file,\
             open(os.path.join(self.expFolder, prefix + '.answer'), 'w', encoding="utf-8") as text_file, \
             open(os.path.join(self.expFolder, prefix + '.span'), 'w', encoding="utf-8") as span_file:
             for i in tqdm.tqdm(indices):
                 contextLine =  linecache.getline(os.path.join(self.dataFolder, sourceTier + '.context'), i+1)
                 questionLine = linecache.getline(os.path.join(self.dataFolder, sourceTier + '.question'), i+1)
                 textLine = linecache.getline(os.path.join(self.dataFolder, sourceTier + '.answer'), i+1)
                 spanLine = linecache.getline(os.path.join(self.dataFolder, sourceTier + '.span'), i+1)
                 contextLine = self.lineToWordIndices( contextLine )
                 while( len(contextLine) < self.maxContextLength ):
                     contextLine.append( self.padNumber )

                 questionLine = self.lineToWordIndices( questionLine )
                 while( len(questionLine) < self.maxQuestionLength ):
                     questionLine.append( self.padNumber )

                 if(len(questionLine) > self.maxQuestionLength or len(contextLine) > self.maxContextLength):
                     raise ValueError

                 textLine = self.lineToWordIndices( textLine )

                 contextArray.append( contextLine )
                 questionArray.append( questionLine )

                 context_file.write( " ".join([str(i) for i in contextLine])+'\n' )
                 question_file.write( " ".join([str(i) for i in questionLine])+'\n' )
                 text_file.write( " ".join([str(i) for i in textLine])+'\n' )
                 span_file.write( spanLine )

        np.save(os.path.join(self.expFolder,prefix+'.contextArray'),np.array(contextArray))
        np.save(os.path.join(self.expFolder,prefix+'.questionArray'),np.array(questionArray))

        return np.array(contextArray), np.array(questionArray)

    def lineToWordIndices(self, line):
        words = line.strip().split()
        indices = []
        for token in words:
            if(token in self.wordIdxLookup):
                indices.append(self.wordIdxLookup[token])
            else:
                self.numUnkWordEncountered += 1
                indices.append(0)
        return indices

    #returns a list of indices into matrix
    def getTrainExIndices(self, sourceTier="train"):
        contextLengths = []
        questionLengths = []
        with open(os.path.join(self.dataFolder,sourceTier+".stats"),'r',encoding='utf-8') as stats_file:
            for line in stats_file:
                line = line.strip().split(',')
                contextLengths.append(int(line[0]))
                questionLengths.append(int(line[1]))
        contextLengths = np.array(contextLengths)
        questionLengths = np.array(questionLengths)
        indices = np.nonzero(contextLengths <= self.maxContextLength)[0]
        qIndices = np.nonzero(questionLengths <= self.maxQuestionLength)[0]
        indices = np.intersect1d( indices, qIndices )
        return indices

    def countTrainingExamples(self, fileName ):
        lineCount = 0
        with open(fileName,"r",encoding="utf-8") as cf:
            for line in cf:
                lineCount += 1
        self.numTrainingExamples = lineCount
        return self.numTrainingExamples
