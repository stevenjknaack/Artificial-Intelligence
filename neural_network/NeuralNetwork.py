from math import *
from csv import *
from io import *
from random import *
from time import time

# Steven Knaack; sjknaack; CS540 Summer 2023; P1

# global vars
HIDDEN_NEURONS = 28
LEARNING_RATE = 0.001
MINI_BATCH_SIZE = 5
EPOCH_NUM = 20
INPUT_FILE_NAME = "network1v5.csv"
OUTPUT_FILE_NAME = "network1v6.csv"

# Pre/post-processing methods
def init_data_matrices(csv_file_name) :
  training_file = open(csv_file_name,"r",newline='')
  training_file_reader = reader(training_file)

  input_matrix = []
  label_matrix = []

  index = 0
  for row in training_file_reader :
    label = row.pop(0)
    if (label == "3") :
      label_matrix.append(0)
    elif (label == "7") :
      label_matrix.append(1)
    else :
      continue

    new_row = [label_matrix[index]]
    for pixelValue in row :
      new_row.append(float(pixelValue) / 255) 
    input_matrix.append(new_row)
    index += 1

  training_file.close()
  return [input_matrix, label_matrix]

def get_test_data(csv_name) :
  testing_file = open(csv_name,"r",newline='')
  testing_file_reader = reader(testing_file)

  test_matrix = []

  for row in testing_file_reader :
    new_row = [0]
    for pixelValue in row :
      new_row.append(float(pixelValue) / 255) 
    test_matrix.append(new_row)

  testing_file.close()
  return test_matrix

def output_network_state(network, filename, network_cost = -1) :
  output = open(filename, "w")

  input_num = str(network.number_input_nodes)
  hidden_num = str(network.number_hidden_nodes)
  output.write(input_num + "," + hidden_num + "," + str(network_cost) 
               + "," + str(LEARNING_RATE) + "," + str(MINI_BATCH_SIZE) + "\n")

  line = ""
  for i in range(len(network.weights_2)) : 
    line += str(network.weights_2[i]) + ","
  line += str(network.bias_2)
  output.write(line + "\n")

  line = ""
  for i in range(len(network.bias_1)) :
    line += str(network.bias_1[i]) 
    if i < len(network.bias_1) - 1 :
      line += ","
  output.write(line + "\n")

  for i in range(len(network.weights_1)) : 
      line = ""
      for j in range(len(network.weights_1[i])) :
        line += str(network.weights_1[i][j])
        if j < len(network.weights_1[0]) - 1 :
          line += ","
      output.write(line + "\n")

  output.close()

def get_network_from_file(filename) :
  data = open(filename, "r")
  data_reader = reader(data)

  row_0 = next(data_reader)

  num_input = int(row_0[0])
  num_hidden = int(row_0[1])

  network = neural_network(num_input, num_hidden)

  row_1 = next(data_reader)
  for i in range(len(row_1) - 1) :
    network.weights_2[i] = float(row_1[i]) 
  network.bias_2 = float(row_1[len(row_1) - 1])

  row_2 = next(data_reader)
  for i in range(len(row_2)) :
    network.bias_1[i] = float(row_2[i]) 

  weights = []
  for row in data_reader : 
    for i in range(len(row)):
      row[i] = float(row[i])
    weights.append(row)
  network.weights_1 = weights

  data.close()
  return network

#Network class
class neural_network :
  def __init__(self, num_nodes_input, num_nodes_hidden) :
    self.number_input_nodes = num_nodes_input
    self.number_hidden_nodes = num_nodes_hidden

    self.weights_1 = []
    self.bias_1 = []

    # first layer weights/biases
    for i in range(self.number_hidden_nodes) :
      weights_into_node = []
      for i in range(self.number_input_nodes) :
        weights_into_node.append((random() * 2) - 1)
      self.weights_1.append(weights_into_node)
      self.bias_1.append((random() * 2) - 1)
    
    # second layer weights/biases
    self.weights_2 = []
    for i in range(self.number_hidden_nodes) :
      self.weights_2.append((random() * 2) - 1)
    self.bias_2 = (random() * 2) - 1

  def activate(self, instance) :
     activations = self.activate_for_training(instance)
     return activations[1]

  def activate_for_training(self, instance) :
     # find activations of first layer
    activations_1 = []

    for i in range(self.number_hidden_nodes) :
      count = self.bias_1[i]
      weights = self.weights_1[i]
      for j in range(self.number_input_nodes) :
        count += weights[j] * instance[j + 1]
      activations_1.append(neural_network.activation_func(count))

    # get activation of output neuron
    count = self.bias_2
    for i in range(self.number_hidden_nodes) :
      count += self.weights_2[i] * activations_1[i]

    return [activations_1, neural_network.activation_func(count)]
    
  def activation_func(x) :
    return 1 / (1 + exp(-x))

  def train(self, training_data, training_data_labels) :
    cost = self.cost(training_data, training_data_labels)
    new_cost = cost + 10

    epoch = 1
    shuffled_training_set = training_data.copy()
    shuffle(shuffled_training_set)
    used_set = []
    while epoch <= EPOCH_NUM and abs(new_cost - cost) >= 0.0001 and new_cost >= 1 :
      t = time() #
      print(str(epoch) + " start: $" + str(round(new_cost,4)) + " " + str(round(new_cost-cost,4))) #
      while len(shuffled_training_set) > 0:
        stoc_training_example = []
        stoc_label = []
        for c in range(MINI_BATCH_SIZE) :
          if len(shuffled_training_set) > 0 :
            example = shuffled_training_set.pop(0)
            stoc_training_example.append(example)
            stoc_label.append(example[0])

            rand_index = randint(0,len(used_set))
            used_set.insert(rand_index, example)
        self.update_weights_bias(stoc_training_example, stoc_label, LEARNING_RATE / sqrt(epoch))
      cost = new_cost
      new_cost = self.cost(training_data, training_data_labels)
      print(str(epoch) + " end: " + str(round(time()-t,2)) + "s")#
      epoch += 1
      shuffled_training_set = used_set
      used_set = []

  def update_weights_bias(self, training_data, training_data_labels, learning_rate) :
    for i in range(len(training_data)):
      instance = training_data[i]
      instance_label = training_data_labels[i]

      activations = self.activate_for_training(instance)
      activations_1 = activations[0]
      activations_2 = activations[1]

      increment_coefficient = learning_rate * (activations_2 - instance_label) * activations_2 * (1 - activations_2)
      
      # update layer 2 weights and biase
      for j in range(self.number_hidden_nodes) :
        self.weights_2[j] -= increment_coefficient * activations_1[j]
      self.bias_1[j] -= increment_coefficient

      # update layer 1 weights and biases
      for j in range(self.number_hidden_nodes) :
        increment = increment_coefficient * self.weights_2[j] * activations_1[j] * (1 - activations_1[j])
        for k in range(self.number_input_nodes) :
          self.weights_1[j][k] -= increment * instance[k + 1]
        self.bias_1[j] -= increment

  def cost(self, test_data, test_data_labels) :
    count = 0
    for i in range(len(test_data)) :
      input_data = test_data[i]
      input_label = input_data[0]

      activation = self.activate(input_data)
      activation = max(0.01, min(0.99, activation))

      count += (input_label - activation) ** 2
    
    return 0.5 * count
    
# problem functions
def train_new() :
  matrices = init_data_matrices("mnist_train.csv")
  input_m = matrices[0]
  label_m = matrices[1]

  brain = neural_network(len(input_m[0]) - 1, HIDDEN_NEURONS)
  brain.train(input_m, label_m)
  output_network_state(brain, OUTPUT_FILE_NAME, brain.cost(input_m,label_m))
  
def train_again() :
  matrices = init_data_matrices("mnist_train.csv")
  input_m = matrices[0]
  label_m = matrices[1]

  brain = get_network_from_file(INPUT_FILE_NAME)
  brain.train(input_m, label_m)
  output_network_state(brain, OUTPUT_FILE_NAME, brain.cost(input_m,label_m))
  
def part2_prob1() :
  #matrices = init_data_matrices("mnist_train.csv")
  #input_m = matrices[0]
  #label_m = matrices[1]

  #brain = neural_network(len(input_m[0]) - 1, HIDDEN_NEURONS)
  #brain.train(input_m, label_m)
  output = open("prob1.txt", "w")

  brain = get_network_from_file(INPUT_FILE_NAME)

  for i in range(len(brain.weights_1[0])) : 
    line = ""
    for j in range(len(brain.weights_1)) :
      line += str(round(brain.weights_1[j][i], 4))
      if (j < len(brain.weights_1) - 1) :
        line += ", "
    output.write(line + "\n")

  line = ""
  for i in range(len(brain.bias_1)) :
    line += str(round(brain.bias_1[i], 4)) 
    if (i < len(brain.bias_1) - 1) :
      line += ", "
  output.write(line + "\n")

def part2_prob2() :
  brain = get_network_from_file(INPUT_FILE_NAME)

  line = ""
  for i in range(len(brain.weights_2)) : 
    line += str(round(brain.weights_2[i], 4)) + ", "
  line += str(round(brain.bias_2,4))
  print(line)

def part2_prob3() :
  brain = get_network_from_file(INPUT_FILE_NAME)

  test_m = get_test_data("test.csv") 

  line = ""
  for i in range(len(test_m)) :
    activation = brain.activate(test_m[i])
    line += str(round(activation, 2))
    if (i < len(test_m) - 1) :
      line += ", "

  print(line)

def part2_prob4() :
  brain = get_network_from_file(INPUT_FILE_NAME)

  test_m = get_test_data("test.csv") 

  line = ""
  for i in range(len(test_m)) :
    activation = round(brain.activate(test_m[i]), 2)
    if activation >= 0.5 :
      line += "1"
    else :
      line += "0"

    if (i < len(test_m) - 1) :
      line += ", "
  
  print(line)

def test_input_output():
  matrices = init_data_matrices("mnist_train.csv")
  input_m = matrices[0]
  label_m = matrices[1]

  brain = neural_network(len(input_m[0]) - 1, HIDDEN_NEURONS)
  print(brain.cost(input_m,label_m))
  output_network_state(brain, OUTPUT_FILE_NAME)
  
  brain1 = get_network_from_file(INPUT_FILE_NAME)
  output_network_state(brain1, "input_output.csv")
  print(brain1.cost(input_m,label_m))

# function calls
# func calls
#train_again()
#part2_prob1()
#part2_prob2()
#part2_prob3()
part2_prob4()
#train_new()
#test_cost()
#test_input_output()