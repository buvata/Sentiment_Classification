import re 
import random
from pyvi import ViTokenizer
from random import shuffle
random.seed(1)

word_sentiment = [ "thích", " không thích", " ko thích", "ko", "khong" ]

def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") 
    line = re.sub(r'\W+', " ", line)
    line = re.sub(r'[0-9]+', " ", line)
    line = line.lower()
    line = ViTokenizer.tokenize(line)
    
    for char in line:
        clean_line += char 

    clean_line = re.sub(' +',' ',clean_line) 
    if clean_line[0] == ' ':
        clean_line = clean_line[1:] 
    return clean_line

def random_deletion(words, p):
    if len(words) == 1:
        return words

	#randomly delete words 
    new_words = []
    for word in words :
        r = random.uniform(0, 1)
        if r > p or word in word_sentiment:
                new_words.append(word)

	#deleting all words, return a random word
    if len(new_words) == 0:
	    rand_int = random.randint(0, len(words)-1)
	    return [words[rand_int]]

    return new_words

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words


def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words


def RD(sentence, p_rd, n_aug=3):
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '' ]
    print(words)
    num_words = len(words)

    augmented_sentences = []

    for _ in range(n_aug):
	    a_words = random_deletion(words, p_rd)
	    augmented_sentences.append(' '.join(a_words))

	# augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    augmented_sentences.append(sentence)

    return augmented_sentences
    

def RS(sentence, p_rs, n_aug=3):
	sentence = get_only_chars(sentence)
	words = sentence.split(' ')
	num_words = len(words)

	augmented_sentences = []
	n_rs = max(1, int(p_rs*num_words))

	for _ in range(n_aug):
		a_words = random_swap(words, n_rs)
		augmented_sentences.append(' '.join(a_words))

	# augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	shuffle(augmented_sentences)

	augmented_sentences.append(sentence)

	return augmented_sentences

if __name__ == "__main__":
   
    text = " hôm nay toi thích  ddi hk :))*678" 
    
    aug_txt1 = RD(text, p_rd=0.3, n_aug=3)
    aug_txt2 = RS(text, p_rs=0.2, n_aug=3)

    print(aug_txt1)
    print(aug_txt2)

