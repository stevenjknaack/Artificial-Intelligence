""" Steven J Knaack; CS540 SUM23; P3 Part 2; naive_bayes.py """

from io import open
from math import log

FAKE_SCRIPT_FILE_NAME = 'script.txt'
PROCESSED_FILE_NAME = 'Processed_Eternal_Sunshine_Script.txt' # Script after processing
SENTENCE_FILE_NAME = 'Sentences.txt'
OUTPUT_FILE_NAME = 'bayes.txt'
NUM_SENTENCES_TO_GEN = 26
NUM_CHARS_PER_SENT = 1000
ALPHABET_LEN = 27
PRIOR_REAL_PROB = 0.92
PRIOR_FAKE_PROB = 1 - PRIOR_REAL_PROB

# Here goes the input methods.

def get_script_string (processed_file_name) :
  in_file = open(processed_file_name, 'r')

  num_lines = 0
  string = ''
  for line in in_file :
    num_lines += 1
    if num_lines > 1 :
      raise ValueError('TOO MANY LINES')
    
    if line.find('  ') != -1 :
      raise ValueError('consecutive spaces at ' + str(line.find('  ')))
    
    if line.find('\n') != -1 :
      raise ValueError('newline at ' + str(line.find('\n')))
  
    string = line

  in_file.close()
  
  return string  

# Here doth lies the output methods.
def output_probabilites (naive_bayes, output_file_name) :
  output_file = open(output_file_name, 'w')

  # Output unigram probabilities.
  string = ''
  sum = 0
  for rounded_prob in naive_bayes._probs0:
    string += str(rounded_prob) + ','
    sum += rounded_prob

  output_file.write(f'Script 0 (Real) ({sum}):\n{string[0:-1]}\n\n')

  string = ''
  sum = 0
  for rounded_prob in naive_bayes._probs1:
    string += str(rounded_prob) + ','
    sum += rounded_prob

  output_file.write(f'Script 1 (Fake) ({sum}):\n{string[0:-1]}\n\n')

  string = ''
  sum = 0
  for rounded_prob in naive_bayes._post0:
    string += str(rounded_prob) + ','
    sum += rounded_prob

  output_file.write(f'Script 0 Post (Real) ({sum}):\n{string[0:-1]}\n\n')

  string = ''
  sum = 0
  for rounded_prob in naive_bayes._post1:
    string += str(rounded_prob) + ','
    sum += rounded_prob

  output_file.write(f'Script 1 Post (Fake) ({sum}):\n{string[0:-1]}\n\n')

# Find you the class of NaiveBayes below.
class NaiveBayes :
  def __init__ (self, script0_file_name, script1_file_name) :
    self.script0_string = get_script_string(script0_file_name)
    self._probs0 = NaiveBayes._find_probabilites(self.script0_string)
    
    self.script1_string = get_script_string(script1_file_name)
    self._probs1 = NaiveBayes._find_probabilites(self.script1_string)

    self._post0 = self._find_posterior_probabilites(self._probs0, PRIOR_REAL_PROB)
    self._post1 = self._find_posterior_probabilites(self._probs1, PRIOR_FAKE_PROB)
    
  def _find_probabilites(script_string) :
    unigram_probs = []
    script_string_len = len(script_string)

    for i in range(ALPHABET_LEN) :
      char = NaiveBayes._index_to_char(i)
      num_char_occur = script_string.count(char)
      char_freq = num_char_occur / script_string_len
      unigram_probs.append(char_freq)

    rounded_probs = []
    sum = 0
    for prob in unigram_probs:
      rounded_prob = round(prob, 4)
      if rounded_prob <= 0.0 :
        rounded_prob = 0.0001
      rounded_probs.append(rounded_prob)
      sum += rounded_prob

    difference = 1 - round(sum,4)
    for i in range(len(rounded_probs)):
      rounded_prob = rounded_probs[i]
      rounded_new = rounded_probs[i] + difference
      if rounded_new > 0.0 and rounded_new < 1.0:
        rounded_probs[i] = round(rounded_new, 4)
        break

    return rounded_probs
  
  def _find_posterior_probabilites(self, probs, prior_prob) :
    post_probs = []

    for i in range(len(probs)) :
      prob = probs[i]
      prob_0 = self._probs0[i]
      prob_1 = self._probs1[i]
      denom = prob_0 * PRIOR_REAL_PROB + prob_1 * PRIOR_FAKE_PROB
      post_prob = (prob * prior_prob) / denom
      post_prob = round(post_prob, 4)
      post_probs.append(post_prob)

    return post_probs

  
  def _index_to_char (index) :
    if index < 0 or index > ALPHABET_LEN - 1 :
      raise IndexError("index " + str(index) + " is out of bounds")
    elif index == 0 :
      return ' '
    else : 
      return chr(index + ord('a') - 1)
    
  def _char_to_index (char) :
    ascii = ord(char)
    
    if (ascii < ord('a') or ascii > ord('z')) and ascii != ord(' ') :
      raise IndexError(f'not valid char for index, char={ascii}')
    elif ascii == ord(' '):
      return 0
    else :
      return ascii - ord('a') + 1
    
  def predict_training_script_from_sentence (self, sentence) :
    false_likelyhood = 0
    real_likelyhood = 0
    for char in sentence :
      if char == '\n' :
        continue
      char_index = NaiveBayes._char_to_index(char)
      fake_post = self._post1[char_index]
      false_likelyhood += log(fake_post)
      real_likelyhood += log(1 - fake_post)

    if false_likelyhood > real_likelyhood :
      guess = 1
    else: 
      guess = 0

    return guess
    
# Here we have problem methods.
def problem_7_8 (naive) :
  output_probabilites(naive, OUTPUT_FILE_NAME)

def problem9 (naive) :
  sentence_file = open(SENTENCE_FILE_NAME,'r')

  string = ''
  for sentence in sentence_file :
    guess = naive.predict_training_script_from_sentence(sentence)
    string += f'{guess},'

  print(string[:-1])
  sentence_file.close()

# Below this comment lies method calls.
naive_mind = NaiveBayes(PROCESSED_FILE_NAME, FAKE_SCRIPT_FILE_NAME)
problem_7_8(naive_mind)
problem9(naive_mind)