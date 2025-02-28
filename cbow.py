import fitz
import numpy as np

doc=fitz.open("corpus.pdf")
print(doc)
#extract the doc 
for page in doc:
    content = page.get_text()
    print(content)
word=[]
corpus=[]
for key in content:
  if key.isalnum():
    word.append(key)
  elif key in '\n\t ' and len(word)>0:
    corpus.append(''.join(word))
    # if key in "\n\t":
    #   corpus.append('\n')
    word=[]
# thsi is the part where the puctuations are added to the corpus 
#   elif key in './\\!?[](,):;"\'#':
#     corpus.append(''.join(word))
#     corpus.append(key)
#     word=[]

# this is the collection all the words and stop words and  ,
print(corpus)

# the stop words in the corpus are is, a, in, through  
# so remoeve the stopwords an convert to lowercase
stopword=['is','a','in','through']
corpus_without_stopword=[]
for word in corpus:
   if word not in stopword:
      corpus_without_stopword.append(word.lower())
print(corpus_without_stopword)

# now create the target and context word
def generate_cbow(text,window_size):
   cbow=[]
   for i, target in enumerate(text):
      context_word=text[max(0,i-window_size):i]+  text[i+1:i+window_size+1]
      if len(context_word) == window_size * 2:
            cbow.append((context_word, target))
   return cbow
# now this has the combination of all the context and targett word with window size 1
cbows=generate_cbow(corpus_without_stopword,1)



# for context_words, target_word in cbows:
#     print(f'Context Words: {context_words}, Target Word: {target_word}')

unique_word=sorted(set(corpus_without_stopword))
# 45 unique words
# convert the word into one-hotencoding
def one_hot_encoding(word,unique_word):
   encoding=[1 if word ==w else 0 for w in unique_word]
   return encoding


# creating the onehot encoding for each word
one_hot_encodings = {word: one_hot_encoding(word, unique_word) for word in unique_word}
# for word, encoding in one_hot_encodings.items():
#    print(word, encoding)




cbow_vector_pair=[([one_hot_encodings[word] for word in context_words], one_hot_encodings[target_word]) for  context_words, target_word in cbows]
# print(cbow_vector_pair)

cbow_vector_pair_sum = [(np.sum(np.stack(context_vectors), axis=0), target_vector) for context_vectors, target_vector in cbow_vector_pair]
# print(cbow_vector_pair_sum)

X_train = np.array([pair[0] for pair in cbow_vector_pair_sum])  # Context vectors (summed)
y_train = np.array([pair[1] for pair in cbow_vector_pair_sum])  # Target word one-hot vectors

embedding_size=3
size_of_vocab=len(unique_word)

w1=np.random.uniform(-1,1,(size_of_vocab,embedding_size))
w2=np.random.uniform(-1,1,(embedding_size,size_of_vocab))

learning_rate = 0.01
epochs = 1000

# convert raw scores (logits) into probabilities.it provides an non linearity 
def softmax(x):
    exp_x = np.exp(x - np.max(x))  
    return exp_x / np.sum(exp_x)



def train_cbow(X, y, W1, W2, learning_rate, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            # Forward pass
            hidden_layer = np.dot(X[i], W1)  # Input to hidden
            output_layer = softmax(np.dot(hidden_layer, W2))  # Hidden to output

            # Compute loss= -sum(y* log(softmax))
            loss = -np.sum(y[i] * np.log(output_layer))  
            total_loss += loss

            # Backpropagation

            error = output_layer - y[i]
            dW2 = np.outer(hidden_layer, error)
            dW1 = np.outer(X[i], np.dot(W2, error))

            # Update weights
            W1 -= learning_rate * dW1
            W2 -= learning_rate * dW2

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    return W1, W2

# Train the model
W1, W2 = train_cbow(X_train, y_train, w1, w2, learning_rate, epochs)

def predict_cbow(context_words, W1, W2, one_hot_encodings, unique_words):
    context_vectors = np.array([one_hot_encodings[word] for word in context_words])
    summed_context = np.sum(context_vectors, axis=0)  # Sum context vectors
    hidden_layer = np.dot(summed_context, W1)
    output_layer = softmax(np.dot(hidden_layer, W2))
    predicted_word_index = np.argmax(output_layer)
    
    return unique_words[predicted_word_index]

# Example Prediction
# context_words = ["artificial", "evolving"] #intelligence
context_words = ["machine", "models"] #learning
# context_words = ["training", "enhances"]# data
# context_words = ["words", "meaning"]# convey
predicted_word = predict_cbow(context_words, W1, W2, one_hot_encodings, unique_word)
print("Predicted word:", predicted_word)


