from math import *
from csv import *
from io import *
from random import random

# Steven Knaack; sjknaack; CS540 Summer 2023; P1


# Preprocessing methods
def init_data_matrices(csv_file_name) :
  training_file = open(csv_file_name,"r",newline='')
  training_file_reader = reader(training_file)

  input_matrix = []
  label_matrix = []

  for row in training_file_reader :
    label = row.pop(0)
    if (label == "3") :
      label_matrix.append(0)
    elif (label == "7") :
      label_matrix.append(1)
    else :
      continue

    new_row = []
    for pixelValue in row :
      new_row.append(float(pixelValue) / 255) 
    input_matrix.append(new_row)

  training_file.close()
  return [input_matrix, label_matrix]

# Regression class
class log_regression_model :
  def __init__(self, num_input_nodes) : 
    self.number_nodes = num_input_nodes

    self.bias = (random() * 2) - 1

    self.weights = []
    for i in range(self.number_nodes) :
      self.weights.append((random() * 2) - 1)

  def activate_regression(self, instance) :
    count = self.bias
    for i in range(self.number_nodes) :
      count += self.weights[i] * instance[i]
    return log_regression_model.activation_function(count)

  def train(self, training_data, training_data_labels) :
    cost = self.cost(training_data, training_data_labels)
    new_cost = cost + 10
    num_iterations = 1

    while abs(new_cost - cost) >= 0.0001 and new_cost >= 1 and num_iterations <= 1000:
      self.update_weights_bias(training_data, training_data_labels, 0.1 / sqrt(num_iterations))
      cost = new_cost
      new_cost = self.cost(training_data, training_data_labels)
      num_iterations += 1

  def update_weights_bias(self, training_data, training_data_labels, learning_rate) :
    for i in range(len(training_data)):
      instance = training_data[i]
      instance_label = training_data_labels[i]

      activation = self.activate_regression(instance)
      increment_coefficient = learning_rate * (activation - instance_label)

      for j in range(len(training_data[i])) :
        self.weights[j] -= increment_coefficient * instance[j]
        self.bias -= increment_coefficient
        

  def cost(self, test_data, test_data_labels) :
    count = 0
    for i in range(len(test_data)) :
      input_data = test_data[i]
      input_label = test_data_labels[i]

      activation = self.activate_regression(input_data)
      activation = max(0.01, min(0.99, activation))
      
      #if input_label == 0 and 1 - activation < 0.01 :
      #  count += 10000
      #elif input_label == 0:
      #  count -= log(1 - activation)
      #elif input_label == 1 and activation < 0.01 :
      #  count += 10000
      #elif input_label == 1 :
       # count -= log(activation)

      count = input_label * log(activation) + (1 - input_label) * log(1 - activation)
      
    return -1 * count

  def activation_function(x) :
    return 1 / (1 + exp(-x))

# Problem methods
def prob1_part1() :
  matrices = init_data_matrices("mnist_train.csv")
  feature_vector = matrices[0][0]
  for i in range(len(feature_vector)) :
    feature_vector[i] = round(feature_vector[i], 2)
  print(feature_vector)

def prob1_part2() :
  matrices = init_data_matrices("mnist_train.csv")
  input_m = matrices[0]
  label_m = matrices[1]

  perceptron = log_regression_model(len(input_m[0]))
  #print(str(perceptron.cost(input_m, label_m)))

  perceptron.train(input_m, label_m)
  #print(str(perceptron.cost(input_m, label_m)))

  for i in range(len(perceptron.weights)) :
    perceptron.weights[i] = round(perceptron.weights[i],4)

  perceptron.bias = round(perceptron.bias, 4)

  print(str(perceptron.weights) + ", " + str(perceptron.bias))

def prob1_part3() :
  matrices = init_data_matrices("mnist_train.csv")
  input_m = matrices[0]
  label_m = matrices[1]

  perceptron = log_regression_model(len(input_m[0]))
  perceptron.train(input_m, label_m)
  
  ### get data
  testing_file = open("test.csv","r",newline='')
  testing_file_reader = reader(testing_file)

  test_matrix = []

  for row in testing_file_reader :
    new_row = []
    for pixelValue in row :
      new_row.append(float(pixelValue) / 255) 
    test_matrix.append(new_row)

  testing_file.close()
  
  ### test data
  test_activation_values = []
  for instance in test_matrix :
    activation = perceptron.activate_regression(instance)
    test_activation_values.append(round(activation,2))
  
  ### print result
  print(test_activation_values)

def prob1_part4() :
  matrices = init_data_matrices("mnist_train.csv")
  input_m = matrices[0]
  label_m = matrices[1]

  perceptron = log_regression_model(len(input_m[0]))
  perceptron.train(input_m, label_m)
  
  ### get data
  testing_file = open("test.csv","r",newline='')
  testing_file_reader = reader(testing_file)

  test_matrix = []

  for row in testing_file_reader :
    new_row = []
    for pixelValue in row :
      new_row.append(float(pixelValue) / 255) 
    test_matrix.append(new_row)

  testing_file.close()
  
  ### test data
  test_activation_values = []
  for instance in test_matrix :
    activation = perceptron.activate_regression(instance)
    #activation = round(activation,2)
    if activation >= 0.5 :
      test_activation_values.append(1)
    else :
      test_activation_values.append(0)
  
  ### print result
  print(test_activation_values)


def test() :
  matrices = init_data_matrices("mnist_train.csv")
  input_m = matrices[0]
  label_m = matrices[1]

  perceptron = log_regression_model(len(input_m[0]))
  print(str(perceptron.cost(input_m, label_m)))

  perceptron.train(input_m, label_m)
  print(str(perceptron.cost(input_m, label_m)))

# Called functions
#test()
#prob1_part1()
#prob1_part2()
#prob1_part3()
prob1_part4()





