""" Steven Knaack; sjknaack; CS540 SU23; P4; parametric_estimator.py """

from io import open
from csv import reader
from math import sqrt

INPUT_FILE_NAME = 'time_series_covid19_deaths_US.csv'
OUTPUT_FILE_NAME = 'cumulative_time_series.txt'
PARAMETER_OUTPUT_NAME = 'parameterized_time_series.txt'

STATE_1 = 'Wisconsin'
STATE_2 = 'Texas'

STATES = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 
          'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 
          'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 
          'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
          'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
          'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 
          'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
          'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
          'West Virginia', 'Wisconsin', 'Wyoming']

DATA_INDICES = None # assigned in get_data_from_input
STATE_INDEX = 'Province_State'
POPULATION_INDEX = 'Population'
POPULATION_INDEX_OFFSET = 11 - 1

# Parameters used: Mean of the time-differenced data, Standard deviation of the time-differenced data, 
# Median of the time-differenced data, Linear trend coefficient of the data, Auto-correlation of the data
PARAMETERS = ['mean', 'standard deviation', 'median', 'linear trend', 'auto_correlation']

def get_data_from_input (input_file_name) :
    input_file = open(input_file_name, 'r', newline='')
    input_reader = reader(input_file)
    
    data_indices_raw = next(input_reader)
    data_indices = {}
    for i in range(len(data_indices_raw)) :
        index = data_indices_raw[i]
        data_indices[index] = i 
    DATA_INDICES = data_indices

    data_m = {}
    state_index = DATA_INDICES[STATE_INDEX]
    population_index = DATA_INDICES[POPULATION_INDEX]
    for row in input_reader :
        state = row[state_index]
        if not (state in STATES) :
            continue
        
        if not data_m.__contains__(state) :
            data_m[state] = [state]
        
        state_data = data_m[state]
        for i in range(population_index, len(row)) :
            value = int(row[i])
            j = i - POPULATION_INDEX_OFFSET
            if j >= len(state_data) :
                state_data.append(value)
            else: 
                state_data[j] += value
        
        data_m[state] = state_data

    input_file.close()
    return data_m

def output_data_from_input(input_file_name, output_file_name) :
    data_m = get_data_from_input(input_file_name)
    output_data_from_matrix(data_m)


def output_data_from_matrix(data_m, output_file_name, start_index = 0) :
    output_file = open(output_file_name, 'w')

    for row in data_m :
        output_file.write(str(data_m[row][start_index:])[1:-1] + '\n')

    output_file.close()

# Data manipulation methods

def get_differenced_data (data_m) :
    differenced_data = {}
    for state in data_m :
        state_data = data_m[state]
        state_differenced = state_data[0:2]
        for i in range(3, len(state_data)) :
            curr = state_data[i]
            prev = state_data[i - 1]
            difference = curr - prev
            state_differenced.append(difference)
        differenced_data[state] = state_differenced
    
    return differenced_data

def get_differenced_mean (state_difference) :
    sum = 0

    for diff in state_difference[2:] :
        sum += diff
    
    raw_mean = sum / len(state_difference[2:])

    return raw_mean

def get_differenced_stand_dev (state_difference, state_mean) :
    values = state_difference[2:]

    sum = 0
    for diff in values :
        diff_minus_mean = diff - state_mean
        sum += diff_minus_mean ** 2
    
    standard_deviation = sum / len(values)
    return sqrt(standard_deviation)

def get_differenced_median (state_difference) :
    sorted_values = state_difference[2:].copy()
    sorted_values.sort()

    median_index = len(sorted_values) / 2
    median_index = int(median_index)

    median = sorted_values[median_index]
    return median

def get_linear_trend (state_data, state_mean) :
    values = state_data[2:]
    
    num_sum = 0
    denom_sum = 0
    T = len(values)
    for i in range(T) :
        t = i + 1
        value_i = values[i]
        val_minus_mean = value_i - state_mean
        coeff = t - (T + 1) / 2
        num_sum += val_minus_mean * coeff

        denom_sum += coeff ** 2
    
    linear_trend = num_sum - denom_sum
    return linear_trend

def get_auto_corr (state_data, state_mean) :
    values = state_data[2:]

    num_sum = 0
    denom_sum = (values[0] - state_mean) ** 2

    for i in range(1, len(values)) :
        curr = values[i]
        prev = values[i - 1]

        curr_minus_mean = curr - state_mean
        prev_minus_mean = prev - state_mean

        num_sum += curr_minus_mean * prev_minus_mean

        denom_sum += curr_minus_mean ** 2

    auto_corr = num_sum / denom_sum

    return auto_corr

def parameterize (state_data, differenced_state) :
    mean = get_differenced_mean(differenced_state)

    stand_dev = get_differenced_stand_dev(differenced_state, mean)

    median = get_differenced_median(differenced_state)

    linear_trend = get_linear_trend(state_data, mean)

    auto_corr = get_auto_corr(state_data, mean)

    state = state_data[0]
    population = state_data[1]

    parameters = [state, population, mean, stand_dev, 
                  median, linear_trend, auto_corr]
    
    return parameters

def get_parameter_matrix (data_m) :
    differenced_m = get_differenced_data(data_m)
    
    parameter_m = {}
    for state in data_m :
        state_data = data_m[state]
        state_diff = differenced_m[state]
        parameters = parameterize(state_data, state_diff)
        parameter_m[state] = parameters

    return rescale_matrix(parameter_m)
    
def rescale_matrix (data_m) :
    rescaled_m = {}
    for key in data_m :
        feature = data_m[key]
        rescaled_feature = feature[:2]
        for i in range(2, len(feature)) :
            value = feature[i]
            data_values = [data_m[state][i] for state in data_m]
    
            minimum = min(data_values)
            maximum = max(data_values)

            range_ = maximum - minimum

            value_diff_from_min = value - minimum

            rescaled_value = round(value_diff_from_min / range_, 8)
            rescaled_feature.append(rescaled_value)
        rescaled_m[key] = rescaled_feature
    
    return rescaled_m

def get_parameter_m_from_input (input_file_name) :
    data_m = get_data_from_input(input_file_name)
    param_m = get_parameter_matrix(data_m)
    return param_m
            
# Problem Methods

def part1 (data_m) :
  print(str(data_matrix[STATE_1][2:])[1:-1] + '\n' + str(data_matrix[STATE_2][2:])[1:-1])

def part2 (data_m) :
    differenced = get_differenced_data(data_m)
    print(str(differenced[STATE_1][2:])[1:-1] + '\n' + str(differenced[STATE_2][2:])[1:-1])

def part4 (p_m) :
    output_data_from_matrix(p_m, PARAMETER_OUTPUT_NAME,2)

# Method calls
#output_data_from_input(INPUT_FILE_NAME, OUTPUT_FILE_NAME)
data_matrix = get_data_from_input(INPUT_FILE_NAME)
parameter_m = get_parameter_matrix(data_matrix)

#part1(data_matrix)
#part2(data_matrix)
part4(parameter_m)


