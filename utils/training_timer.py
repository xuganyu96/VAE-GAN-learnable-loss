# A method that counts the total amount of time needed to train all epochs
# by reading through the lines of the README.md file in each of the 
# results directory

def train_timer(README_path):
    
    # Split the string of the readme file by line breaker
    readme_lines = open(README_path, 'r').read().split('\n')
    
    # Meaningful lines are the ones that start with 'Epoch'
    training_lines = [line for line in readme_lines if len(line) > 0 and line[0]=='E']
    
    # Print the number of items in training lines to show
    # that I have the right number of lines
    # print(len(training_lines))
    
    # For each of the training lines, split by space, then take the last time
    # this is definitely the "decimal" value of time used to train each epoch
    times = [float(line.split(' ')[-1]) for line in training_lines]
    
    # return the sum of all entries to get the total time used for training
    return sum(times)