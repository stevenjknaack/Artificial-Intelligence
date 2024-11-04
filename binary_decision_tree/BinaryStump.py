from io import open
from csv import reader
from math import log2

### Steven Knaack; sjknaack; CS540 SUM23; P2 BinaryStump.py

# Feature and Label indices (after removind ID from index 0)
CLUMP_THICKNESS = 0
UNIFORMITY_CELL_SIZE = 1
UNIFORMITY_CELL_SHAPE = 2 
MARGINAL_ADHESION = 3
SINGLE_EPITHELIEL_CELL_SIZE = 4
BARE_NUCLEI = 5
BLAND_CHROMATIN = 6
NORMAL_NUCLEI = 7
MITOSIS = 8
CLASS_LABEL = 9 

# other constants
INPUT_FILE_NAME = 'breast-cancer-wisconsin.csv'
FEATURE_INDEX = UNIFORMITY_CELL_SHAPE # indexed 4 on the problem page, 2 here
BENIGN = 2
MALIGNANT = 4

VALUE_INDEX = 0 
LABEL_INDEX = 1

# input methods
def get_feature_matrix(csv_filename) :
    input_file = open(csv_filename, 'r', newline='')
    input = reader(input_file)

    feature_m = []
    for row in input:
        row.pop(0)
        #feature_m.append(row) # must convert to ints first
        feature_m.append([int(row[FEATURE_INDEX]), int(row[CLASS_LABEL])])
    
    return feature_m

# output methods

# primary methods/classes
def get_value_count(feature_m, feature_index) :
    count = {}

    for feature in feature_m :
        value = feature[feature_index]
        count[value] = count.setdefault(value, 0) + 1

    return count

def entropy(feature_m, event_index):
    value_count = get_value_count(feature_m, event_index)

    total_features = len(feature_m)

    accum = 0
    for value in value_count:
        probability = value_count[value] / total_features
        accum -= probability * log2(probability)
    
    return accum

def get_subset_features(feature_m, index, value) :
    subset = []
    for feature in feature_m :
        curr = feature[index]
        if (curr == value) :
            subset.append(feature)
    return subset

def specific_conditional_entropy(feature_m, event_index, prior_index, prior_value) :
    feature_subset = get_subset_features(feature_m, prior_index, prior_value)

    return entropy(feature_subset, event_index)

def conditional_entropy(feature_m, event_index, prior_index) :
    total_features = len(feature_m)
    value_count = get_value_count(feature_m, prior_index)

    accum = 0
    for value in value_count:
        probability = value_count[value] / total_features
        accum -= probability * specific_conditional_entropy(feature_m, event_index, prior_index, value)

    return accum
        
def info_gain(feature_m, event_index, prior_index) :
    return entropy(feature_m, event_index) - conditional_entropy(feature_m, event_index, prior_index)

def split_around(feature_m, feature_index, feature_value) :
    split = [[],[]]
    for feature in feature_m :
        if feature[feature_index] <= feature_value : 
            split[0].append(feature)
        else :
            split[1].append(feature)
    
    return split

def conditional_entropy_for_split(feature_m, event_index, prior_index, prior_value) :
    total_features = len(feature_m)

    split = split_around(feature_m, prior_index, prior_value)
    total_neg = len(split[0])
    total_pos = len(split[1])

    neg_values = get_value_count(split[0], event_index)
    pos_values = get_value_count(split[1], event_index)

    accum = 0
    for value in neg_values :
        count = neg_values[value]
        probability = count / total_neg
        coeff = count / total_features
        accum -= coeff * log2(probability)

    for value in pos_values :
        count = pos_values[value]
        probability = count / total_pos
        coeff = count / total_features
        accum -= coeff * log2(probability)

    return accum


class binary_stump:
    def __init__(self, feature_m, feature_index, label_index) : 
        values = get_value_count(feature_m, feature_index).keys()

        min_con_split = 1000
        min_con_split_value = None
        for value in values :
            con_split = conditional_entropy_for_split(feature_m, label_index, feature_index, value)
            if con_split < min_con_split :
                min_con_split = con_split
                min_con_split_value = value
        
        self.split_value = min_con_split_value


if __name__ == '__main__':
    # problem methods
    feature_m = get_feature_matrix(INPUT_FILE_NAME)
    stump = binary_stump(feature_m, VALUE_INDEX, LABEL_INDEX)

    def p1_q1() :
        print(get_value_count(feature_m, LABEL_INDEX))

    def p1_q2() :
        print(str(round(entropy(feature_m, LABEL_INDEX), 4)))

    def p1_q3() :
        split = split_around(feature_m, VALUE_INDEX, stump.split_value)

        neg_value_count = get_value_count(split[0], LABEL_INDEX)
        pos_value_count = get_value_count(split[1], LABEL_INDEX)

        print(str(neg_value_count[BENIGN]) + ',' + str(pos_value_count[BENIGN]) + ',' + str(neg_value_count[MALIGNANT]) + ',' + str(pos_value_count[MALIGNANT]))


    def p1_q4() :
        print(round(entropy(feature_m, LABEL_INDEX) - conditional_entropy_for_split(feature_m, LABEL_INDEX, VALUE_INDEX, stump.split_value), 4))


    # method calls
    p1_q1() 
    p1_q2()
    p1_q3()
    p1_q4()


