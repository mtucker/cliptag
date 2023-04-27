from nltk.corpus import brown
from nltk import FreqDist
from nltk import pos_tag
from nltk.tag.mapping import map_tag
from nltk import download as nltk_download

nltk_download('brown')
nltk_download('averaged_perceptron_tagger')
nltk_download('universal_tagset')

WORDS_FILE = "./features/words.txt"
UNSUPPORTED_POS_TAGS = ['ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'PRON', '.', 'X']
TOP_K = 200000

print("Extracting words from the Brown corpus...")
words = brown.tagged_words(tagset='universal')

# print("Part-of-speech tagging words...")
# tagged_words = pos_tag(words, tagset='universal')

print("Filtering words...")
filtered_words = [word for word, pos in words if pos not in UNSUPPORTED_POS_TAGS]

print("Calculating word frequency distribution...")
word_freqdist = FreqDist(filtered_words)

print("Getting the most common words...")
common_words = [word for word, _ in word_freqdist.most_common(TOP_K)]

# for word, p in word_freqdist.most_common(TOP_K):
#   print(f"{word} {p}")

# for word in filtered_words[:100]:
#   print(f"{word}")

print("Writing common words to file...")
with open(WORDS_FILE, 'w') as f:
    f.write("\n".join(common_words))