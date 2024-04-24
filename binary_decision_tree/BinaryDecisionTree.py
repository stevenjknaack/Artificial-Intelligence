from io import open
from csv import reader
from math import log2
from random import randint

### Steven Knaack; sjknaack; CS540 SUM23; P2 BinaryStump.py

# Feature and Label indices (after removind ID from index 0)
SAMPLE_CODE_NUMBER = 0
CLUMP_THICKNESS = 1
UNIFORMITY_CELL_SIZE = 2
UNIFORMITY_CELL_SHAPE = 3 
MARGINAL_ADHESION = 4
SINGLE_EPITHELIEL_CELL_SIZE = 5
BARE_NUCLEI = 6
BLAND_CHROMATIN = 7
NORMAL_NUCLEOLI = 8
MITOSIS = 9
CLASS_LABEL = 10 

BENIGN = 2
MALIGNANT = 4

LEFT = 0
RIGHT = 1
ROOT = 2

INTERNAL = 0
LEAF = 1

TRAIN1 = 0
TRAIN2 = 1
PRUNE = 2

# other constants
INPUT_FILE_NAME = 'breast-cancer-wisconsin.csv'
OUTPUT_FILE_NAME = 'tree1.txt'

FEATURE_INDEXES = [BLAND_CHROMATIN, CLUMP_THICKNESS, 
                   MARGINAL_ADHESION, NORMAL_NUCLEOLI,
                     UNIFORMITY_CELL_SHAPE, UNIFORMITY_CELL_SIZE] 

FEATURE_INDEXES.sort()

VALUE_INDEXES = range(len(FEATURE_INDEXES)) 
LABEL_INDEX = len(FEATURE_INDEXES)

TRAIN_TYPE = TRAIN1
MAX_DEPTH = 6

# input methods
def get_feature_matrix(csv_filename) :
    input_file = open(csv_filename, 'r', newline='')
    input = reader(input_file)

    feature_m = []
    for row in input:
        if len(row) != 11 :
            continue

        feature_v = []
        for feature_index in FEATURE_INDEXES :
            feature_value = row[feature_index]

            if feature_value == '?' :
                feature_value = randint(1,10)

            feature_v.append(int(feature_value))

        feature_v.append(row[CLASS_LABEL])

        feature_m.append(feature_v)

    input_file.close()
    
    return feature_m

def get_test_matrix(csv_filename) :
    input_file = open(csv_filename, 'r', newline='')
    input = reader(input_file)

    feature_m = []
    for row in input:
        if len(row) != 10 :
            continue

        feature_v = []
        for feature_index in FEATURE_INDEXES :
            feature_value = row[feature_index]

            if feature_value == '?' :
                feature_value = randint(1,10)

            feature_v.append(int(feature_value))

        feature_m.append(feature_v)

    input_file.close()
    
    return feature_m

# output methods
def output_readable_tree(tree, output_filename) :
    output = open(output_filename, 'w')

    nodes_to_visit = [tree.root]
    level_stack = [0]
    while len(nodes_to_visit) > 0 :
        node = nodes_to_visit.pop(len(nodes_to_visit) - 1)
        level = level_stack.pop(len(level_stack) - 1)
        
        if node.side == RIGHT :
                output.write('\n' + ' ' * (level - 1) + 'else')
        
        if node.role == LEAF :
            output.write(' return ' + str(node.classification))
        else :
            nodes_to_visit.append(node.right_child)
            level_stack.append(level + 1)

            nodes_to_visit.append(node.left_child)
            level_stack.append(level + 1)
        
            feat_abs_ind = FEATURE_INDEXES[node.split_feature_i] + 1
            threshold = node.split_feature_value

            if_string = f'if (x{feat_abs_ind} <= {threshold})'
            if node.role == ROOT: 
                output.write(if_string)
            else :
                output.write('\n' + ' ' * level + if_string)


    output.close()

# primary methods/classes
def get_value_count(feature_m, feature_index) :
    count = {}

    #i = 0
    for feature in feature_m :
        #print(feature, feature_index, i)
        #i += 1
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

def get_best_split_index(feature_m, feature_indexes, label_index) :
    min_con_entropy = 1000
    min_con_value = None

    for feature_ind in feature_indexes:
        con_entropy = conditional_entropy(feature_m, label_index, feature_ind)
        if con_entropy < min_con_entropy :
            min_con_entropy = con_entropy
            min_con_value = feature_ind

    return feature_ind

def get_best_split_value(feature_m, feature_index, label_index) : 
    values = get_value_count(feature_m, feature_index).keys()

    min_con_split = 1000
    min_con_split_value = None
    for value in values :
        con_split = conditional_entropy_for_split(feature_m, label_index, feature_index, value)
        if con_split < min_con_split :
            min_con_split = con_split
            min_con_split_value = value
    
    return (min_con_split_value, entropy(feature_m, label_index) - min_con_split)

class binary_node() :
    def __init__(self, role, side) :
        self.role = role
        self.side = side

        self.left_child = None
        self.right_child = None

        self.split_feature_i = None
        self.split_feature_value = None

        self.classification = None

class binary_decision_tree :
    def __init__(self, feature_m, feature_indexes, label_index, train_type, max_depth = 0) :
        self.root = binary_node(ROOT, None)

        if train_type == TRAIN1 :
            self.train_tree(feature_m, feature_indexes, label_index)
        elif train_type == TRAIN2 :
            self.train_tree_2(feature_m, feature_indexes, label_index)
        elif train_type == PRUNE :
            self.train_tree_prune(feature_m, feature_indexes, label_index, max_depth)
        else :
            print('ERROR')

    def best_info_gain(self, feature_m, feature_indexes, label_index) :
        best_info_gain = -1
        best_gain_index = -1
        best_gain_value = None

        for feat_ind in feature_indexes :
            best_split = get_best_split_value(feature_m, feat_ind, label_index)
            feature_info_gain = best_split[1]
            if feature_info_gain > best_info_gain :
                best_info_gain = feature_info_gain
                best_gain_index = feat_ind
                best_gain_value = best_split[0]
        
        return (best_gain_index, best_gain_value, best_info_gain)
    
    def train_tree(self, feature_m, feature_indexes, label_index) :
        node_stack = [self.root]
        subset_stack = [feature_m]

        while len(node_stack) > 0 :
            node = node_stack.pop(len(node_stack) - 1)
            feat_subset = subset_stack.pop(len(subset_stack) - 1)

            best_index_value = self.best_info_gain(feat_subset, feature_indexes, label_index)
            node.split_feature_i = best_index_value[0]
            node.split_feature_value = best_index_value[1]
            best_info_gain = best_index_value[2]

            if best_info_gain == 0 :
                value_count = get_value_count(feat_subset, label_index)

                max_count = -1
                max_value = None
                for value in value_count :
                    count = value_count[value]
                    if count > max_count :
                        max_count = count
                        max_value = value
                
                node.role = LEAF
                node.classification = max_value
                continue

            split_set = split_around(feat_subset, node.split_feature_i, node.split_feature_value)
            under_set = split_set[0]
            over_set = split_set[1]
            
            under_count = get_value_count(under_set, label_index)
            under_len = len(under_count)
            over_count = get_value_count(over_set, label_index)
            over_len = len(over_count)
           # print(len(under_set), len(over_set))

            if under_len == 0 and over_len == 1:
                node.role = LEAF
                node.classification = over_set[0][label_index]
                continue

            if over_len == 0 and under_len == 1 :
                node.role = LEAF
                node.classification = under_set[0][label_index]
                continue

            if over_len == 1 :
                single_classification = over_set[0][label_index]
                new_node = binary_node(LEAF, RIGHT)
                new_node.classification = single_classification
                node.right_child = new_node
            else :
                node.right_child = binary_node(INTERNAL, RIGHT)
                node_stack.append(node.right_child)
                subset_stack.append(over_set)

            if under_len == 1 :
                single_classification = under_set[0][label_index]
                new_node = binary_node(LEAF, LEFT)
                new_node.classification = single_classification
                node.left_child = new_node
            else :
                node.left_child = binary_node(INTERNAL, LEFT)
                node_stack.append(node.left_child)
                subset_stack.append(under_set)

    def train_tree_2(self, feature_m, feature_indexes, label_index) :
        node_stack = [self.root]
        subset_stack = [feature_m]

        while len(node_stack) > 0 :
            node = node_stack.pop(len(node_stack) - 1)
            feat_subset = subset_stack.pop(len(subset_stack) - 1)

            node.split_feature_i = get_best_split_index(feat_subset, feature_indexes, label_index)
            node.split_feature_value = get_best_split_value(feat_subset, node.split_feature_i, label_index)

            split_set = split_around(feat_subset, node.split_feature_i, node.split_feature_value)
            under_set = split_set[0]
            over_set = split_set[1]
            
            under_count = get_value_count(under_set, label_index)
            under_len = len(under_count)
            over_count = get_value_count(over_set, label_index)
            over_len = len(over_count)
           # print(len(under_set), len(over_set))

            if  under_len == 0 and over_len == 1 :
                node.role = LEAF
                node.classification = over_set[0][label_index]
                continue

            if under_len == 1 and over_len == 0: 
                node.role = LEAF
                node.classification = under_set[0][label_index]
                continue

            if over_len == 1 :
                single_classification = over_set[0][label_index]
                new_node = binary_node(LEAF, RIGHT)
                new_node.classification = single_classification
                node.right_child = new_node
            else :
                node.right_child = binary_node(INTERNAL, RIGHT)
                node_stack.append(node.right_child)
                subset_stack.append(over_set)

            if under_len == 1 :
                single_classification = under_set[0][label_index]
                new_node = binary_node(LEAF, LEFT)
                new_node.classification = single_classification
                node.left_child = new_node
            else :
                node.left_child = binary_node(INTERNAL, LEFT)
                node_stack.append(node.left_child)
                subset_stack.append(under_set)

    def train_tree_prune(self, feature_m, feature_indexes, label_index, max_depth) :
        node_stack = [self.root]
        subset_stack = [feature_m]
        depth_stack = [0]

        while len(node_stack) > 0 :
            node = node_stack.pop(len(node_stack) - 1)
            feat_subset = subset_stack.pop(len(subset_stack) - 1)
            depth = depth_stack.pop(len(depth_stack) - 1)

            if depth >= max_depth :
                value_count = get_value_count(feat_subset, label_index)

                max_count = -1
                max_value = None
                for value in value_count :
                    count = value_count[value]
                    if count > max_count :
                        max_count = count
                        max_value = value
                
                node.role = LEAF
                node.classification = max_value
                continue


            best_index_value = self.best_info_gain(feat_subset, feature_indexes, label_index)
            node.split_feature_i = best_index_value[0]
            node.split_feature_value = best_index_value[1]

            split_set = split_around(feat_subset, node.split_feature_i, node.split_feature_value)
            under_set = split_set[0]
            over_set = split_set[1]
            
            under_count = get_value_count(under_set, label_index)
            under_len = len(under_count)
            over_count = get_value_count(over_set, label_index)
            over_len = len(over_count)
           # print(len(under_set), len(over_set))

            if under_count == 0 :
                node.role = LEAF
                node.classification = over_set[0][label_index]
                continue

            if over_count == 0 :
                node.role = LEAF
                node.classification = under_set[0][label_index]
                continue

            if over_len == 1 :
                single_classification = over_set[0][label_index]
                new_node = binary_node(LEAF, RIGHT)
                new_node.classification = single_classification
                node.right_child = new_node
            else :
                node.right_child = binary_node(INTERNAL, RIGHT)
                node_stack.append(node.right_child)
                subset_stack.append(over_set)
                depth_stack.append(depth + 1)

            if under_len == 1 :
                single_classification = under_set[0][label_index]
                new_node = binary_node(LEAF, LEFT)
                new_node.classification = single_classification
                node.left_child = new_node
            else :
                node.left_child = binary_node(INTERNAL, LEFT)
                node_stack.append(node.left_child)
                subset_stack.append(under_set)
                depth_stack.append(depth + 1)

    
    def classify(self, feature_v) :
        node = self.root   
    
        while node.role != LEAF :
            feat_ind = node.split_feature_i
            threshold = node.split_feature_value

            if feature_v[feat_ind] <=  threshold:
                node = node.left_child
            else :
                node = node.right_child
            
        return int(node.classification)
            
# problem methods
feature_m = get_feature_matrix(INPUT_FILE_NAME)
tree = binary_decision_tree(feature_m, VALUE_INDEXES, LABEL_INDEX, TRAIN_TYPE, MAX_DEPTH)

def p2_q1() :
    output_readable_tree(tree, OUTPUT_FILE_NAME)
    #print(feature_m)

def p2_q2() :
    pass

def p2_q3() :
    test_m = get_test_matrix('test.csv')
    
    classifications = []
    for feature in test_m :
        classification = tree.classify(feature)
        classifications.append(classification)
    
    print(classifications)

def p2_q4() :
    pass

# method calls
p2_q1() 
#p2_q2()
p2_q3()
#p2_q4()


