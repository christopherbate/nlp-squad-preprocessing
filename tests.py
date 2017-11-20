import unittest
import ExperimentSetup
import spacy
import squad_processer
import VocabBuilder
import EMBuilder
import numpy as np

class TestSQAUADProcesser(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load('en')
        self.sp = squad_processer.SQUADProcesser(self.nlp.tokenizer, data_dir="tests")

    def test_break_file(self):
        self.sp.break_file('dev',filename='dev-v1.1.json',countExamples = True)
        count = 0
        with open('tests/dev.context', "r", encoding='utf-8') as cf:
            for line in cf:
                count += 1
        self.assertEqual( count, 10570, "Dev context file incorrect size." )
        self.assertEqual( count, self.sp.numTrainExamples, "Counted training examples doesn't equal num of lines in context file.")

        statsLineCount = 0
        with open('tests/dev.stats', "r",encoding='utf-8') as stats_file:
            for line in stats_file:
                statsLineCount += 1
        self.assertEqual(statsLineCount, count, "Stats file line count does not equal context file.")

class TestVocabBuilder(unittest.TestCase):
    def setUp(self):
        self.vocabBuilder = VocabBuilder.VocabBuilder( ['tests/train.context','tests/train.question'], 'tests/vocab.dat' )
        self.vocabBuilder.createVocab()
        self.vocabBuilder.saveVocab()

    def test_create(self):
        self.assertTrue(len(self.vocabBuilder.word_list) > 1)
        self.assertTrue(len(self.vocabBuilder.word_list) == len(self.vocabBuilder.word_idx_lookup))
        self.assertEqual(len(self.vocabBuilder.word_list), 104492, "Word list length not 105032" )
        self.assertEqual( self.vocabBuilder.word_list[1],"the")
        #self.assertEqual( self.vocabBuilder.word_list[-1],"Origenists")

    def test_load(self):
        count = 0
        self.vocabBuilder.loadVocab()
        self.assertEqual(len(self.vocabBuilder.word_list), 104492)
        self.assertEqual(len(self.vocabBuilder.word_list),
                            len(self.vocabBuilder.word_idx_lookup))
        self.assertEqual( self.vocabBuilder.word_list[1],"the")
        #self.assertEqual( self.vocabBuilder.word_list[-1],"Origenists")

class TestMatrixBuilder(unittest.TestCase):
    def setUp(self):
        self.vb = VocabBuilder.VocabBuilder( ['tests/train.context', 'tests/train.question'],'tests/vocab.dat')
        self.vb.loadVocab()
        self.em = EMBuilder.EmbeddingMatrix(self.vb.word_idx_lookup, gloveSize='100',
                                            vocabFile='tests/vocab.dat',saveLoc='tests/embedding_matrix')

    def test_create_and_save(self):
        self.em.buildEmbeddingMatrix()
        loadedArray = np.load('tests/embedding_matrix100.npz')['matrix']
        self.assertEqual(self.em.num_matched, 82275)
        self.assertEqual(loadedArray.shape,self.em.embedding_matrix.shape)

    def test_load(self):
        self.em.loadEmbeddingMatrix()
        self.assertEqual(self.em.embedding_matrix.shape[0],104492)



class ExperimentSetupTest(unittest.TestCase):
    def setUp(self):
        self.vb = VocabBuilder.VocabBuilder( ['tests/train.context', 'tests/train.question'],'tests/vocab.dat')
        self.vb.loadVocab()
        self.exp = ExperimentSetup.ExperimentSetup(self.vb.word_idx_lookup,
            wordVecSize='100',maxContextLength=300,maxQuestionLength=10,padZero=True,
            experimentFolder='tests/exp',dataFolder='tests', train_percent=0.9,
            shuffleExamples=True, sourceTier="dev")

    def getTrainIndices_test(self):
        indices = self.exp.getTrainExIndices(sourceTier="teststats")
        self.assertEqual(tuple(indices),tuple([0,1,3,4,8]))

    def test_creation(self):
        contextArray,questionArray = self.exp.generateExperimentFiles(self.exp.trainIndices,
                                                    "dev.idx","dev")
        count = 0
        ciFile = 'tests/exp/dev.idx.context'
        with open( ciFile, "r", encoding='utf-8' ) as cif:
            for line in cif:
                count+=1


        self.assertEqual(contextArray.shape, (self.exp.finalTrainExCount,
                                              self.exp.maxContextLength))

        self.assertEqual(count,self.exp.finalTrainExCount)

    def test_line_to_word_idx(self):
        testLine = "the the the the"
        indices = self.exp.lineToWordIndices( testLine )
        self.assertEqual( indices, [1,1,1,1] )

class TestFullRun(unittest.TestCase):
    def fullRun(self):
        nlp = spacy.load('en')
        print("Splitting files.")
        sp = squad_processer.SQUADProcesser(nlp.tokenizer,data_dir='tests')
        sp.break_file("dev",sp.devFile,True)
        print("Building vocab.")
        vb = VocabBuilder.VocabBuilder(['tests/train.context','tests/train.question'],
                                        'tests/vocab.dat')
        vb.createVocab()
        vb.saveVocab()
        vb.loadVocab()

        print("Building embedding.")
        em = EMBuilder.EmbeddingMatrix( vb.word_idx_lookup, gloveDir="tests",
                                        vocabFile='tests/vocab.dat',
                                        saveLoc='tests/embedding_matrix')
        em.buildEmbeddingMatrix()

        es = ExperimentSetup.ExperimentSetup(vb.word_idx_lookup,
                wordVecSize='100',maxContextLength=300,maxQuestionLength=20,padZero=True,
                experimentFolder='tests/exp',dataFolder='tests',train_percent=0.8,
                shuffleExamples=False,sourceTier="dev")

        ca, qa = es.generateExperimentFiles(es.trainIndices,"train.idx","dev")
        es.generateExperimentFiles(es.xvalIndices,"xval.idx","dev")

        self.assertEqual( ca.shape, (es.finalTrainExCount,300))
        self.assertEqual( qa.shape, (es.finalTrainExCount,20))


if __name__ == '__main__':
    unittest.main()
