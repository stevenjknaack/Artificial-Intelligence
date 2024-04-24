""" Steven J Knaack; CS540 SUM23; P3 Part 1; MarkovChain.py """

from io import open
from re import sub
from random import random

# Global Parameters
SCRIPT_FILE_NAME = 'Eternal_Sunshine_Script.txt' # Movie Script for 'Eternal Sunshine of the Spotless Mind'
PROCESSED_FILE_NAME = 'Processed_Eternal_Sunshine_Script.txt' # Script after processing
SENTENCE_FILE_NAME = 'Sentences.txt'

OUTPUT_PROB_FILE_NAME = 'Probabilities.txt'

NUM_SENTENCES_TO_GEN = 26
NUM_CHARS_PER_SENT = 1000

# Global Constants
ALPHABET_LEN = 27

# Input / Preprocessing methods
def process_script_file (script_file_name, output_file_name) :
  """
  Take a movie script and process it into the output file.
  Processing includes: 
    deleting all new lines and other non-space whitespace,
    keeping only letters and single spaces,
    making all characters lowercase,
    replacing all consecutive spaces with only a single space.
  """
  in_file = open(script_file_name, 'r')
  out_file = open(output_file_name, 'w')

  for line in in_file:
    string = sub('\n+',' ', line)
    string = sub('[^A-Za-z \s]+','', string)
    string = sub('\s+',' ', string)
    string = string.lower()
    string = string.strip()

    if string == '':
      continue

    out_file.write(' ' + string)

  in_file.close()
  out_file.close()

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

# Output Methods
def output_probabilites (markov_chain, output_file_name) :
  output_file = open(output_file_name, 'w')

  # Output unigram probabilities.
  rounded_probs = []
  sum = 0
  for prob in markov_chain._unigram_probs:
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

  string = ''
  sum = 0
  for rounded_prob in rounded_probs:
    string += str(rounded_prob) + ','
    sum += rounded_prob

  output_file.write(f'Unigram ({sum}):\n{string[0:-1]}\n\n')

  # Output Bigram no lapace probabilites.
  rounded_probs = []
  for a in range(len(markov_chain._bigram_nl_probs)) :
    a_prior_probs = markov_chain._bigram_nl_probs[a]
    sum = 0
    a_rounded = []
    for prob in a_prior_probs :
      rounded_prob = round(prob, 4)
      a_rounded.append(rounded_prob)
      sum += rounded_prob
    rounded_probs.append(a_rounded)

    difference = 1 - round(sum, 4)
  
    for b in range(len(a_prior_probs)) :
      rounded_prob = a_prior_probs[b]
      rounded_new = round(rounded_prob + difference, 4)
      if rounded_new > 0.0 and rounded_new < 1.0 and b != 0:
        rounded_probs[a][b] = rounded_new
        break

  string = ''
  sum = 0
  for a_rounded in rounded_probs :
    for ab_rounded in a_rounded :
      string += str(ab_rounded) + ','
      sum += ab_rounded
    string = string[0:-1] + "\n"

  output_file.write(f'Bigram with no Lapace ({round(sum,6)}):\n{string}\n\n')

  # Output Bigram with lapace probabilites.
  rounded_probs = []
  for a in range(len(markov_chain._bigram_probs)) :
    a_prior_probs = markov_chain._bigram_probs[a]
    sum = 0
    a_rounded = []
    for prob in a_prior_probs :
      rounded_prob = round(prob, 4)
      if rounded_prob <= 0.0 :
        rounded_prob = 0.0001
      a_rounded.append(rounded_prob)
      sum += rounded_prob
    rounded_probs.append(a_rounded)

    difference = 1 - round(sum, 4)
  
    for b in range(len(a_prior_probs)) :
      rounded_prob = a_prior_probs[b]
      rounded_new = round(rounded_prob + difference, 4)
      if rounded_new > 0.0 and rounded_new < 1.0 and b != 0:
        rounded_probs[a][b] = rounded_new
        break

  string = ''
  sum = 0
  for a_rounded in rounded_probs :
    for ab_rounded in a_rounded :
      string += str(ab_rounded) + ','
      sum += ab_rounded
    string = string[0:-1] + "\n"

  output_file.write(f'Bigram ({round(sum,6)}):\n{string}\n\n')


  output_file.close()

# Main classes/methods
class MarkovChain :
  def __init__ (self, script_file_name) :
    self.script_string = get_script_string(script_file_name)

    self._script_string_len = len(self.script_string)

    self._unigram_probs = self._find_unigram_probabilites()

    self._bigram_nl_probs = self._find_bigram_nl_probabilites()

    self._bigram_probs = self._find_bigram_probabilites()

    self._trigram_probs = self._find_trigram_probabilites()

  def _find_unigram_probabilites(self) :
    unigram_probs = []

    for i in range(ALPHABET_LEN) :
      char = MarkovChain._index_to_char(i)
      num_char_occur = self.script_string.count(char)
      char_freq = num_char_occur / self._script_string_len
      unigram_probs.append(char_freq)

    return unigram_probs
  
  def _find_bigram_nl_probabilites (self) :
    bigram_probs = []

    for a in range(ALPHABET_LEN) :
      a_prior_probs = []
      char = MarkovChain._index_to_char(a)
      count_a = self.script_string.count(char)
      for b in range(ALPHABET_LEN) :
        if count_a == 0 :
          ab_bi_prob = 0
        else : 
          char_2 = MarkovChain._index_to_char(b)
          substr = char + char_2
          count_ab = self.script_string.count(substr)
          ab_bi_prob = count_ab / count_a
        a_prior_probs.append(ab_bi_prob)
      bigram_probs.append(a_prior_probs)
  
    return bigram_probs

  def _find_bigram_probabilites(self) :
    bigram_probs = []

    for a in range(ALPHABET_LEN) :
      a_prior_probs = []
      char = MarkovChain._index_to_char(a)
      count_a = self.script_string.count(char)
      for b in range(ALPHABET_LEN) :
        char_2 = MarkovChain._index_to_char(b)
        substr = char + char_2
        count_ab = self.script_string.count(substr)
        ab_bi_prob = (count_ab + 1) / (count_a + ALPHABET_LEN)
        a_prior_probs.append(ab_bi_prob)
      bigram_probs.append(a_prior_probs)

    rounded_probs = []
    for a in range(len(bigram_probs)) :
      a_prior_probs = bigram_probs[a]
      sum = 0
      a_rounded = []
      for prob in a_prior_probs :
        rounded_prob = round(prob, 4)
        if rounded_prob <= 0.0 :
          rounded_prob = 0.0001
        a_rounded.append(rounded_prob)
        sum += rounded_prob
      rounded_probs.append(a_rounded)

    difference = 1 - sum
  
    for b in range(len(a_rounded)) :
      rounded_prob = a_rounded[b]
      rounded_new = round(rounded_prob + difference, 4)
      if rounded_new > 0.0 and rounded_new < 1.0 and b != 0:
        rounded_probs[a][b] = rounded_new
        break
  
    return rounded_probs

  def _find_trigram_probabilites(self) :
    trigram_probs = []

    for a in range(ALPHABET_LEN) :
      a_prior_probs = []
      char1 = MarkovChain._index_to_char(a)
  
      for b in range(ALPHABET_LEN) :
        ab_prior_probs = []
        char2 = MarkovChain._index_to_char(b)
        prior_chars = char1 + char2
        count_ab = self.script_string.count(prior_chars)

        for c in range(ALPHABET_LEN) :
          char3 = MarkovChain._index_to_char(c)
          full_sub_str = prior_chars + char3
          count_abc = self.script_string.count(full_sub_str)
          abc_prob = (count_abc + 1) / (count_ab + ALPHABET_LEN)
          ab_prior_probs.append(abc_prob)
        
        a_prior_probs.append(ab_prior_probs)
      
      trigram_probs.append(a_prior_probs)

    rounded_probs = []
    for a in range(len(trigram_probs)) :
      a_prior_probs = trigram_probs[a]
      sum = 0
      a_rounded = []
      for b in range(len(a_prior_probs)) :
        ab_prob = a_prior_probs[b]
        ab_rounded = []
        for abc_prob in ab_prob :
          rounded_prob = round(abc_prob, 4)
          if rounded_prob <= 0.0 :
            rounded_prob = 0.0001
          sum += rounded_prob
          ab_rounded.append(rounded_prob)
        
        difference = 1 - sum

        for c in range(len(ab_rounded)) :
          rounded_prob = ab_rounded[c]
          rounded_new = round(rounded_prob + difference, 4)
          if rounded_new > 0.0 and rounded_new < 1.0 and c != 0:
            ab_rounded[c] = rounded_new
            break
        
        a_rounded.append(ab_rounded)
      rounded_probs.append(a_rounded)

    return rounded_probs

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
      raise IndexError('not valid char for index')
    elif ascii == ord(' '):
      return 0
    else :
      return ascii - ord('a') + 1
    
  def generate_sentence (self, num_char, start_char_index) :   
    second_last_char = MarkovChain._index_to_char(start_char_index)
    
    second_last_char_index = MarkovChain._char_to_index(second_last_char)
    distribution = self._bigram_probs[second_last_char_index]
    last_char = MarkovChain._gen_char_from_distribution(distribution)

    sentence = second_last_char + last_char

    for i in range(num_char - 2) :
      second_last_char_index = MarkovChain._char_to_index(second_last_char)
      last_char_index = MarkovChain._char_to_index(last_char)
      nl_bigram = self._bigram_nl_probs[second_last_char_index][last_char_index]
      if nl_bigram < 0 :
        raise ValueError('something is wrong with bigram matrix')
      elif nl_bigram == 0 :
        distribution = self._bigram_probs[last_char_index]
      else : 
        distribution = self._trigram_probs[second_last_char_index][last_char_index] 
      
      second_last_char = last_char
      last_char = MarkovChain._gen_char_from_distribution(distribution)
      sentence += last_char

    return sentence
  
  def _gen_char_from_distribution (CDF_array) :
    rand = round(random(), 4)
    lower_bound = 0

    char = ''
    for i in range(len(CDF_array)) :
      upper_bound = round(lower_bound + CDF_array[i], 4)

      if rand >= lower_bound and rand < upper_bound :
        char = MarkovChain._index_to_char(i)

      lower_bound = upper_bound

    if char == '' :
      rand = int(random() * 27)
      char = MarkovChain._index_to_char(rand)
      #sum = 0
      #for i in CDF_array :
        #sum += i
      #raise RuntimeError(f'no char generated on rand={rand}, upperbound={upper_bound}, sum={sum},cdf={CDF_array}')

    return char

# Problem Methods
def process_script () :
  process_script_file(SCRIPT_FILE_NAME, PROCESSED_FILE_NAME)

markov_chain = MarkovChain(PROCESSED_FILE_NAME)

def question2 () :
  output_probabilites(markov_chain, OUTPUT_PROB_FILE_NAME)
  print(markov_chain._trigram_probs)

def question () :
  sentence_file = open(SENTENCE_FILE_NAME, 'w')
  for i in range(ALPHABET_LEN - 1) :
    sentence_file.write(markov_chain.generate_sentence(NUM_CHARS_PER_SENT, i + 1) + '\n')
  sentence_file.close()

# Method Calls
#process_script()
#question2()
question()


