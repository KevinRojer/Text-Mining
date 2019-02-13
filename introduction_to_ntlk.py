import nltk
import re
import string
from pprint import pprint
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.draw.tree import TreeView

text = '''Another ex-Golden Stater, Paul Stankowski from Oxnard, is contending for a berth on the U.S. Ryder Cup team after winning his first PGA Tour event last year and staying within three strokes of the lead through three rounds of last month's U.S. Open. H.J. Heinz Company said it completed the sale of its Ore-Ida frozen-food business catering to the service industry to McCain Foods Ltd. for about $500 million. It's the first group action of its kind in Britain.'''
print(text)

# Sentence splitting
nltk_sentence_splitted = sent_tokenize(text)
for index, sentence in enumerate(nltk_sentence_splitted, 1):
    print(f'SENTENCE {index}: {sentence}')

# Tokenization
example_sentence = "I'll refuse to permit you to obtain the refuse permit."
tokenized = nltk.word_tokenize(example_sentence)
print(tokenized)

# Part of speech tagging
pos_tagged = nltk.pos_tag(tokenized)
print(pos_tagged)

# Remove stop words
english_stopwords = stopwords.words('english')
set_english_stopwords = set(english_stopwords) # sets are faster to check if an element is in
print(english_stopwords)

without_stopwords = []
for token in tokenized:
    if token not in set_english_stopwords:
        without_stopwords.append(token)

print(without_stopwords)

# Cleaning up the text
messy_sentence = "The point of this example is to _learn how basic text cleaning works_ on *very simple* data."
tokenized_messy_sentence = nltk.word_tokenize(messy_sentence)
table = {ord(char): '' for char in string.punctuation} # in case you're interested, this is called a dict comprehension

cleaned_messy_sentence = []
for token in tokenized_messy_sentence:
    
    cleaned_word = token.translate(table) # the translate method allows us to remove all unwanted charachters
    cleaned_messy_sentence.append(cleaned_word)

print(cleaned_messy_sentence)

# Stemming and Lemmatization
porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

porterlemmas = []
wordnetlemmas = []
snowballlemmas = []

for word in tokenized:
    porterlemmas.append(porter.stem(word))
    snowballlemmas.append(snowball.stem(word))
    wordnetlemmas.append(wordnet.lemmatize(word))

print('Porter')
print(porterlemmas)
print('Snowball')
print(snowballlemmas)
print('Wordnet')
print(wordnetlemmas)

# Named Entity Recognition (NER)
text = '''In August, Samsung lost a US patent case to Apple and was ordered to pay its rival $1.05bn (£0.66bn) in damages for copying features of the iPad and iPhone in its Galaxy range of devices. Samsung, which is the world's top mobile phone maker, is appealing the ruling. A similar case in the UK found in Samsung's favour and ordered Apple to publish an apology making clear that the South Korean firm had not copied its iPad when designing its own devices.'''
sentences = nltk.sent_tokenize(text)
for sentence in sentences:
    tokens = nltk.word_tokenize(sentence)
    tokens_pos_tagged = nltk.pos_tag(tokens)
    tokens_pos_tagged_and_named_entities = ne_chunk(tokens_pos_tagged)
    print()
    print('ORIGINAL SENTENCE', sentence)
    print('NAMED ENTITY RECOGNITION OUTPUT', tokens_pos_tagged_and_named_entities)
    
    
  # Constituency/dependency parsing  
  constituent_parser = nltk.RegexpParser('''
NP: {<DT>? <JJ>* <NN>*} # NP
P: {<IN>}           # Preposition
V: {<V.*>}          # Verb
PP: {<P> <NP>}      # PP -> P NP
VP: {<V> <NP|PP>*}  # VP -> V (NP|PP)*''')
  
tokens = ['In', 'the', 'house', 'the', 'yellow', 'cat', 'saw', 'the', 'dog']
tagged = nltk.pos_tag(tokens)
print(tagged)
constituent_structure = constituent_parser.parse(tagged)
print(constituent_structure)
constituent_structure

# Save tree to file
TreeView(constituent_structure)._cframe.print_to_file('output.pdf')
