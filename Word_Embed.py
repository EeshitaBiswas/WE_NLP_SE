import os
import gensim
# from gensim import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# define training data
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = MySentences('/Users/eeshitabiswas/Desktop/Prelims/Resources_2')

# train model
model = gensim.models.Word2Vec(sentences, min_count=1)

# summarize vocabulary
words = list(model.wv.vocab)

#print(words)
#access vector for one word
print(model['darmstadtium'])
#print(model['dysprosium'])

# save model
model.wv.save('aa.bin')

w1 = "darmstadtium"
print("Similar By Word: {0}".format(w1), model.wv.similar_by_word(w1))

w1 = "darmstadtium"
print("Most similar to: {0}".format(w1), model.wv.most_similar(positive=w1))

'''w1 = "dysprosium"
print("Similar By Word: {0}".format(w1), model.wv.similar_by_word(w1))

w1 = "dysprosium"
print("Most similar to: {0}".format(w1), model.wv.most_similar(positive=w1))'''

# similarity between two identical words
print("Similarity between 'darmstadtium' and 'darmstadtium'", model.wv.similarity(w1="darmstadtium", w2="darmstadtium"))

