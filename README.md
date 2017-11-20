Contains processing code for SQuAD dataset.

See main.py for an example processing pipeline.

Description of classes.

squad_processer.py:
  - Loads raw JSON and splits the data into multiple files.
  - One file set for training, one file set for testing data.
  - Each training example is broken up into context, question, answer, and span.
  - The above four items are each placed in the context, question, answer, and span
    files.
  - Data is saved as space-seperated tokens generated from JSON data.
    - Tokenizer is the Spacy tokenizer.
  - Thus, each training/testing example corresponds to a line number shared accross
    these files.
  - Saves the data in the ./data subdirectory.

VocabBuilder.py:
  - Given a set of input files (default: training context, training questions),
    builds a vocab sorted by word frequency (createVocab), saves it as a sorted word_list
    (saveVocab), and allows you to load a vocab (loadVocab).
  - Creating or loading a vocab creates member variables word_list, which is
    simply a list of words, and word_idx_lookup, which is a dictionary with words
    as keys and indices into word_list as keys.
  - The word at index zero is always "_UNK_" for padding purposes.

EMBuilder.py:
  - Loads the requested GLoVE matrix (50, 100, 200, or 300 dimensional)
  - Creates a dictionary of all words in the GLoVE file with words as keys
    and Numpy vector as values.
  - Builds a matrix of embedding vectors. Vectors are rows in the matrix, organized
    in the same order as words appear in the supplied vocab file (output of VocabBuilder).
  - Any words in the supplied vocab that do not appear in the GloVE file will be Given
    a vector of all zeros.
  - Save the resulting matrix as a np array (buildembeddingMatrix)
  - Load the embedding matrix (loadEmbeddingMatrix)

ExperimentSetup.py
  - The purpose of the ExperimentSetup class is to load previously generated
    vocab, embedding matrix, and split data files, and generate as output Numpy
    arrays of training data which can easily be fed into RNN models, saved, and
    reproduced using the class.
  - Constructor: allows you to pass in the settings for your experiment
    - Train/Xval split percentage
    - Whether to zero-pad sequences
    - The max context and question lengths
    - Whether to shuffle training examples
    - What folder to save the training output in.
  - The constructor will then sort through the training examples, select only
    those with the context and question lengths within the limit, and then shuffle
    indices to those questions and save the indicices as "trainIndices" and "xvalIndices"
  - generateExmperimentFiles method takes a set of indices as input, and then
    generates a set of experiment files (context, question, answer, span) split into
    training and xval sets. Instead of saving as tokens, the files contain indicies
    into the embedding matrix for every word. Examples are zero-padded as necessary.
  - At the end of generateExperimentFiles, the question and context arrays are then
    written as numpy arrays for easy loading.
