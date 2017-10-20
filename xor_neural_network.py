import math
import copy
from random import randint

def activate_function(x):

    y = 1/(1 + math.exp(-x))

    return y

def activate_all(values):

    for i in range(len(values)):
        values[i] = activate_function(values[i])

    return values

def forward_propagate(correct_input, paths):

    hidden_layer = [0, 0, 0]
    path_count = 0

    current_output = 0

    for current_path in paths:

        if path_count == 0:

            for path in current_path:

                m = path[0]
                start = path[1]
                dest = path[2]

                hidden_layer[dest] += (correct_input[start]*m)

            hidden_layer = activate_all(hidden_layer)

            path_count +=1


        else:

            for path in current_path:

                m = path[0]
                start = path[1]
                dest = path[2]

                current_output += (hidden_layer[start]*m)

            return [paths,current_output, hidden_layer, correct_input]

def find_training_sum(current_x, correct_x):

    f = activate_function(current_x)

    derivate = derivate_sigmoid(f)
    diff = correct_x - f

    return derivate*diff

def derivate_sigmoid(x):

    return x*(1 - x)

def divide_vectors(v1, v2):
    v3 = []

    for i in range(len(v1)):
        for j in range(len(v2)):
            v3.append(v1[i]*v2[j])
    return v3


def backwards_propagate(input,paths, correct_x, current_x, hidden_layer):

    tuner = 1

    #Find the training sum
    training_sum = find_training_sum(current_x, correct_x)*tuner

    #Make a copy to hold our new data structure
    new_paths = copy.deepcopy(paths)
    #Calculate new values for the weights going from the last hidden layer to the output node

    for i in range(len(new_paths)):
        if(i == 0):
            #Input to hidden

            delta_hidden_sums = [0]*len(paths[len(paths) - 1])

            for j in range(len(delta_hidden_sums)):
                delta_hidden_sums[j] = (training_sum/paths[len(paths) - 1][j][0])*derivate_sigmoid(hidden_layer[j])

            delta_weights = divide_vectors(delta_hidden_sums,input)


            for x in range(len(paths[0])):
                new_paths[0][x][0] = paths[0][x][0] + delta_weights[x]

        else:
            #Hidden to ouput
            path = paths[i]
            new_path = new_paths[i]

            for j in range(len(path)):
                new_path[j][0] = path[j][0] + (training_sum/hidden_layer[path[j][1]])

    return new_paths

#Neural network for solving xor problem

def train_neural_network():

    #Set the initial weights and paths (This is a kind of stupid way to do it)
    #First element in innermost list is the weight, second is the start node in current layer (from top to bottom), third is the destination node in the next layer

    paths = [[[0.8,0,0],[0.4,0,1],[0.3,0,2], [0.2,1,0], [0.9,1,1],[0.5,1,2]] ,[[0.3,0,0],[0.5,1,0] ,[0.9,2,0]]]

    #All possible cominations of xor for two variables

    inputs = [[0,0],[0,1],[1,1],[1,0]]
    outputs = [0,1,0,1]

    #How many times should the network be pruned?

    runs = 10000

    values = []

    for i in range(runs):

        equation_num = randint(0,len(inputs) - 1)

        #Forward propagate through the graph

        values = forward_propagate(inputs[equation_num], paths)

        #Receive output from forward propagating, put into backwards_propagate

        paths = backwards_propagate(inputs[equation_num],values[0],outputs[equation_num],values[1],values[2])
    print("Ending error: " + str(outputs[equation_num] - activate_function(values[1])))


train_neural_network()
