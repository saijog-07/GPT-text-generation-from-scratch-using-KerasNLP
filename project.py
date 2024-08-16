# basic imports

# To interact with operating system and to read or write the files from the local system.
import os

# It is a library in keras which is used to perfrom tasks related nlp.
import keras_nlp

# keras is a deep learning api used to implement neural networks
import keras

# tensorflow is a framework used to build ML and DL models 

# data is a submodel used for data preprocessing
import tensorflow.data as tf_data

# strings is a submodel used for performing string related operations like string manipulation.
import tensorflow.strings as tf_strings

# setting up the parameters

# Data parameters : Following are the three 3 data parameters which we are going to use

# This parameter defines the number of samples that will be passed through the model at once during training. A batch size of 64 means that the model will process 64 data points before updating the model parameters. Larger batch sizes can lead to more stable training but require more memory.

BATCH_SIZE = 64

# This parameter sets a minimum length for input strings. Strings shorter than 512 characters will be discarded. This ensures that the model only trains on sufficiently long sequences, which can be important for certain NLP tasks.

MIN_STRING_LEN = 512

# This is the length of training sequences in tokens. Each input sequence to the model will be 128 tokens long. If sequences are shorter than this, they may be padded, and if they are longer, they may be truncated.

SEQ_LEN = 128

# Following 5 parameters are model parameters.

# This parameter defines the size of the embedding vectors. Each token in the vocabulary will be represented by a 256-dimensional vector in the embedding space.

EMBED_DIM = 256

# This paramter determines the size of the intermediate or hidden layers in the feedforward network of each transformer block.
# EMBED_DIM is the size of the input and output layers and this paramters is the size of hidden layer.

FEED_FORWARD_DIM = 128

# This specifies the number of attention heads in the multi-head attention mechanism. Having multiple heads allows the model to focus on different parts of the sequence simultaneously, capturing various aspects of the input data.

NUM_HEADS = 3

# This indicates the number of layers (or blocks) in the transformer model. Each layer consists of a multi-head attention mechanism followed by a feedforward network.

NUM_LAYERS = 2

# This limits the size of the vocabulary to 5000 tokens. This parameter helps to control the number of parameters in the model, as a larger vocabulary would increase the size of the embedding matrix and potentially the complexity of the model.

VOCAB_SIZE = 5000

# Following is a training parameter

# This parameter defines the number of complete passes through the entire training dataset. Training for more epochs can improve model performance, but it also increases the risk of overfitting (Performing a little too well for the training data but might not perform well for test or valid data) if the model trains for too long.

EPOCHS = 5

# Following is an inferece parameter

#This specifies the number of tokens the model should generate during the inference phase. This is relevant for tasks like text generation, where the model produces a sequence of tokens based on a given input prompt.

NUM_TOKENS_TO_GENERATE = 80




# In the following code we are using TensorFlow's Keras utilities to download a dataset, extract it, and then load it for further processing. 

# using Tensorflow's Keras to download and extract the dataset.


keras.util.get_file(
    origin="https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip",
    extract=True,
)

# keras.util.get_file() function is used to download the file from the given dataset. Origin parameter consists of the URL and as the file is in Zip format, we need to extract it so the extract parameter is set to true.

dir = os.path.expanduser("~/.keras/datasets/simplebooks/")

# The above line set a directory where the downloaded and extracted dataset will reside

# Loading the train dataset

raw_train_ds = (
    tf_data.TextLineDataset(dir + "simplebooks-92-raw/train.txt")
    .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

# tf_data.TextLineDataset: Creates a dataset of lines from the specified text file.
# filter: Filters out lines with length less than MIN_STRING_LEN.
# batch: Groups the lines into batches of size BATCH_SIZE.
# shuffle: Shuffles the dataset with a buffer size of 256.

raw_val_ds = (
    tf_data.TextLineDataset(dir + "simplebooks-92-raw/valid.txt")
    .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
    .batch(BATCH_SIZE)
)

# similar to the training data, validation data is also loaded.

# Computing the vocabulary / Train the tokenizer

vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    raw_train_ds,
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,
    reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
)

# keras_nlp.tokenizers.compute_word_piece_vocabulary: This function computes the vocabulary for a WordPiece tokenizer.
# It analyzes the input dataset to determine the most frequent tokens and their subword components.

# raw_train_ds: This is the training dataset used to compute the vocabulary.
# vocabulary_size=VOCAB_SIZE: Specifies the size of the vocabulary.

# lowercase=True: Indicates whether to convert all tokens to lowercase before tokenization.
# This is helpful to standardize tokenization and reduce vocabulary size.

# reserved_tokens=["[PAD]", "[UNK]", "[BOS]"]: Specifies a list of reserved tokens.

# [PAD]: Used to pad sequences to the same length.
# [UNK]: Represents out-of-vocabulary (OOV) tokens.
# [BOS]: Represents the beginning of a sequence.


# Load the tokenizer

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
    lowercase=True,
)

# keras_nlp.tokenizers.WordPieceTokenizer: This is the WordPiece tokenizer class provided by the Keras NLP library.
# vocabulary=vocab: Specifies the vocabulary used by the tokenizer.
# sequence_length=SEQ_LEN: Defines the maximum sequence length for tokenized sequences.
# lowercase=True: Indicates whether to convert all tokens to lowercase before tokenization.


# Tokenizing the data

# The following code adds [BOS] i.e. beginning of sequence token at the start of every sequence and it also ensures that the sequence length is constant i.e. 128. It sequence length is less then it gets padded and if the sequence length 

# keras_nlp.layers.StartEndPacker: This is a layer provided by Keras NLP (Natural Language Processing) toolkit. It is used to pack sequences with special tokens at the start (and optionally at the end) of sequences to a fixed length.

start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)

# preprocessor function:

# tokenizer(inputs) : It takes the raw text data as an input and converts it into sequences of tokens using the tokenizer and stores the tokens in the outputs.
# start_packer(outputs) : It takes the sequences of tokens as an input and converts it into the sequences of equal lengths by appending the special token at the beginning of each sequence.
# The sequences of tokens are also stored in labels which will be used in further training of the model because these are the primary building blocks for training the model.
# After preprocessing the data it returns the features(preprocessed data) and labels i.e. the sequences of tokens.


def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels


# The following code preprocesses the training data one element at a time, but num_parallel_calls = tf_data.AUTOTUNE allows to preprocess multiple elements at the same time i.e. it enables parallel preprocessing. And it also helps the tensorflow to identify optimal number of parallel processings to conduct so that the task will become faster.

# prefetch() function fetches the next batch of data in advance while the current batch of data is getting preprocessed. It acts as a buffer to ensure model has data to process all the time.
# similar to the map method tf_data.AUTOTUNE tells us how many batches to fetch in advance.

train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(
    tf_data.AUTOTUNE
)
val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(
    tf_data.AUTOTUNE
)

# as a result of this, we will obtain preprocessed and efficiently loaded training dataset and valid datasets which are ready for further model training process.



# Process of building the model :

# In the following code, construction and compilation of transformer based sequence model will take place with the help of keras.

# A transformer-based sequence model is a type of neural network architecture specifically designed to handle sequential data, such as natural language text. 

inputs = keras.layers.Input(shape=(None,), dtype="int32")

# keras.layers.Input: This creates an input layer for the model.
# shape=(None,): The input shape is a sequence of arbitrary length. The None dimension allows sequences of varying lengths.
# dtype="int32": The data type of the input is integer, suitable for token IDs.



# After forming an input layer, we will form an embedding layer.
# For that, we will be using keras_nlp.layers.TokenAndPositionEmbedding() because it combines token embeddings with positional embeddings.

# token embeddings convert discrete tokens which are in the form of words or subwords into continuoues vectors of fixed size which allows model to understand the tokens.

# Positional emeddings provide information about the order of tokens in a given sequence.

embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=SEQ_LEN,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)

# The embedding layer will take the input of size of vocabulary, sequence length, and the dimension of vector embedding along with mask_zero = True, meaning while embedding the tokens it would ignore the tokens with id = 0 i.e. padding tokens.

x = embedding_layer(inputs)

# x = embedding_layer(inputs): Applies the embedding layer to the inputs, producing the embedded representation of the input sequence.


# Now we will form a series of decoder layers.

for _ in range(NUM_LAYERS):
    decoder_layer = keras_nlp.layers.TransformerDecoder(
        num_heads=NUM_HEADS,
        intermediate_dim=FEED_FORWARD_DIM,
    )
    x = decoder_layer(x)

# for _ in range(NUM_LAYERS): This loop adds multiple Transformer decoder layers to the model.
# keras_nlp.layers.TransformerDecoder: Defines a Transformer decoder layer.
# num_heads=NUM_HEADS: The number of attention heads in the multi-head attention mechanism.
# Each attention head processes input sequence in parallel with the other attention heads.
# intermediate_dim=FEED_FORWARD_DIM: The dimensionality of the feed-forward network within the decoder.
# x = decoder_layer(x): Each decoder layer is applied to the output of the previous layer. Providing a single argument (x) means the decoder will skip cross-attention and only perform self-attention.

# Now we will form an output layer.

outputs = keras.layers.Dense(VOCAB_SIZE)(x)

# keras.layers.Dense: A dense (fully connected) layer.
# outputs = keras.layers.Dense(VOCAB_SIZE)(x): Applies the dense layer to the final output of the last decoder layer. This layer generates logits for each token in the vocabulary.
# logits are unnormalized form of the output of all the layers combined.


# Definition of the model

model = keras.Model(inputs=inputs, outputs=outputs)

# keras.Model: This creates a Keras model by specifying the input and output layers.
# inputs: The input layer defined at the beginning.
# outputs: The output layer defined above.


# specifying a loss function

#  It quantifies how well or poorly the model is performing by measuring the difference between the model's predictions and the actual target values. 

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# SparseCategoricalCrossentropy: A loss function used for multi-class classification problems where the target is an integer. from_logits=True indicates that the model outputs raw logits.


# specifying a metric to evaluate the model

perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)

# keras_nlp.metrics.Perplexity: Perplexity is a common metric for evaluating language models, representing the model's ability to predict the next token in a sequence.
# mask_token_id=0: Indicates that the metric should ignore padding tokens.


# compilation of the model

model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])

# model is compiler with popular optimizer 'adam' along with loss function and metric.

model.summary()

# Now we will train the model using fit() method.

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# The "packer" layers adds the [BOS] token for us.
prompt_tokens = start_packer(tokenizer([""]))
prompt_tokens

def next(prompt, cache, index):
    logits = model(prompt)[:, index - 1, :]
    # Ignore hidden states for now; only needed for contrastive search.
    hidden_states = None
    return logits, hidden_states, cache


# Initialize Greedy Sampler: Sets up a greedy sampler to generate text.
# Generate Tokens: Uses the sampler to generate a sequence of tokens starting from the given prompt, choosing the highest probability token at each step.
# Detokenize: Converts the generated tokens back into readable text.
# Print Output: Displays the generated text.

sampler = keras_nlp.samplers.GreedySampler()
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,  # Start sampling immediately after the [BOS] token.
)
txt = tokenizer.detokenize(output_tokens)
print(f"Greedy search generated text: \n{txt}\n")

# Initialize Beam Sampler: Sets up a beam sampler with 10 beams for generating text.
# Generate Tokens: Uses the beam sampler to generate a sequence of tokens starting from the # given prompt, exploring multiple sequences and keeping the top 10 (beams) at each step.
# Detokenize: Converts the generated tokens back into readable text.
# Print Output: Displays the generated text.

sampler = keras_nlp.samplers.BeamSampler(num_beams=10)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Beam search generated text: \n{txt}\n")

# Initialize Random Sampler: Sets up a random sampler for generating text.
# Generate Tokens: Uses the random sampler to generate a sequence of tokens starting from the given prompt, selecting tokens randomly based on their probability distribution at each step.
# Detokenize: Converts the generated tokens back into readable text.
# Print Output: Displays the generated text.

sampler = keras_nlp.samplers.RandomSampler()
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Random search generated text: \n{txt}\n")


# Initialize Top-K Sampler: Sets up a Top-K sampler with k=10 for generating text.
# Generate Tokens: Uses the Top-K sampler to generate a sequence of tokens starting from the given prompt, selecting the next token randomly from the top 10 most probable tokens at each step.
# Detokenize: Converts the generated tokens back into readable text.
# Print Output: Displays the generated text.

sampler = keras_nlp.samplers.TopKSampler(k=10)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-K search generated text: \n{txt}\n")


# Initialize Top-P Sampler: Sets up a Top-P (nucleus) sampler with p=0.5 for generating text.
# Generate Tokens: Uses the Top-P sampler to generate a sequence of tokens starting from the given prompt, selecting the next token randomly from the smallest set of top tokens whose cumulative probability is at least 0.5.
# Detokenize: Converts the generated tokens back into readable text.
# Print Output: Displays the generated text.

sampler = keras_nlp.samplers.TopPSampler(p=0.5)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-P search generated text: \n{txt}\n")





#Define Callback Class: A TopKTextGenerator class is defined, inheriting from keras.callbacks.Callback.

# Initialization: The class initializes a TopKSampler with k (number of top tokens to sample from).
# on_epoch_end Method: This method is called at the end of each training epoch. It generates text using the Top-K sampler and prints it.
# Generate Text: Inside on_epoch_end:

# Sampler: Uses the Top-K sampler to generate tokens from the given prompt.
# Detokenize: Converts the generated tokens back into readable text.
# Print Output: Displays the generated text.
# Create Callback Instance: An instance of TopKTextGenerator is created with k=10.

# Training Loop: The model is trained using model.fit, with the text_generation_callback included in the callbacks list.

# Dummy Training Loop: Demonstrates the callback functionality by running for 2 epochs on a subset of the training dataset.
# In summary, this code defines a callback to generate and print text using Top-K sampling at the end of each training epoch.

class TopKTextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model using top-k."""

    def __init__(self, k):
        self.sampler = keras_nlp.samplers.TopKSampler(k)

    def on_epoch_end(self, epoch, logs=None):
        output_tokens = self.sampler(
            next=next,
            prompt=prompt_tokens,
            index=1,
        )
        txt = tokenizer.detokenize(output_tokens)
        print(f"Top-K search generated text: \n{txt}\n")


text_generation_callback = TopKTextGenerator(k=10)
# Dummy training loop to demonstrate callback.
model.fit(train_ds.take(1), verbose=2, epochs=2, callbacks=[text_generation_callback])







