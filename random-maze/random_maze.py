"""
Steven Knaack
P5
CS540 SU23
"""

from io import open
from random import randint, shuffle
from io import open
from math import sqrt

# global parameters and variables

HEIGHT = 55
WIDTH = 47

VWALL = 0
HWALL = 1
INT = 2
FREE = 3
TRAVELED = 4
START = 5
GOAL = 6
BROKEN_VWALL = 7
BROKEN_HWALL = 8
TRAVELED_VWALL = 9

UP = 'U'
DOWN = 'D'
LEFT = 'L'
RIGHT = 'R'

CHARS = ['|', '--', '+', '  ', '@@', '  ', '  ', ' ', '  ', '@'] # TODO CHANGE '-' to '--'

BFS = 0
DFS = 1

MANHATTEN = 0
EUCLIDEAN = 1

UNSEARCHED = 0
SEARCHED = 1

OUPUT_FILENAME = 'output.txt'

# primary classes and methods
class RandomMaze :
    def __init__(self, height, width) :
        # generate maze
        self.cells = RandomMaze._generate_random_maze(height, width)
        self.successors = RandomMaze._generate_successor_m(self.cells)
        self.start = (0, int(len(self.successors[0]) / 2))
        self.goal = (len(self.successors) - 1, int(len(self.successors[len(self.successors) - 1]) / 2))
    
    def _generate_random_maze(height, width) :
        cells = []
        visited = []

        # initialize grids
        for i in range(height * 2 + 1) :
            row = []
            visited_row = []
            for j in range(width * 2 + 1) :
              if i == 0 and j == width :
                  row.append(START)
              elif i == height * 2 and j == width :
                  row.append(GOAL)
              elif i % 2 == 0 and j % 2 == 1 :
                  row.append(HWALL) 
              elif i % 2 == 0 :
                  row.append(INT) 
              elif i % 2 == 1 and j % 2 == 0 :
                  row.append(VWALL)
              else :
                  row.append(FREE)

              visited_row.append(False)
            cells.append(row) 
            visited.append(visited_row)

        # get random starting free cell
        row = None
        col = None
        not_found = True
        
        while not_found:
            row = randint(0, len(cells) - 1)
            col = randint(0, len(cells[0]) - 1)
            if cells[row][col] == FREE :
                not_found = False

        #recursive backtrack
        RandomMaze._recursive_backtrack(cells, row, col, visited)

        return cells
    
    def _recursive_backtrack(cells, row, col, visited) :
        stack = [(row, col, [])]
        while len(stack) > 0 :
            # get curr context
            row, col, actions = stack.pop()

            # mark this cell as visited 
            visited[row][col] = True

            # shuffle possible actions
            if len(actions) == 0 :
                actions = [UP, DOWN, LEFT, RIGHT]
                shuffle(actions)

            # find next action
            while len(actions) > 0 :
                # get cell based on action
                action = actions.pop()
                action_i = None
                action_j = None

                if action == UP :
                    action_i = row - 2
                    action_j = col
                elif action == DOWN :
                    action_i = row + 2
                    action_j = col
                elif action == LEFT :
                    action_i = row
                    action_j = col - 2
                elif action == RIGHT : 
                    action_i = row
                    action_j = col + 2       
                else :
                    raise(ValueError('bruh'))   

                # determine if valid
                in_bounds = action_i > 0 and action_i < len(cells) - 1
                in_bounds = in_bounds and action_j > 0 and action_j < len(cells[0]) - 1

                if in_bounds and not visited[action_i][action_j] :
                    # break wall
                    if action == UP :
                        cells[row - 1][col] = BROKEN_HWALL
                    elif action == DOWN :
                        cells[row + 1][col] = BROKEN_HWALL
                    elif action == LEFT :
                        cells[row][col - 1] = BROKEN_VWALL
                    elif action == RIGHT : 
                        cells[row][col + 1] = BROKEN_VWALL
                    else :
                        raise(ValueError('bruh 2'))

                    # push next context
                    if len(actions) > 0 :
                        stack.append((row, col, actions))

                    stack.append((action_i, action_j, []))

                    break
    
    def _generate_successor_m(cells) :
        successors = []
        for i in range(1, len(cells) - 1, 2) :
            success_row = []

            for j in range(1, len(cells[i]) - 1, 2) :
                success_el = []

                if not i == 1 and cells[i - 1][j] == BROKEN_HWALL : # UP
                    success_el.append(UP)

                if not i == len(cells) - 2 and cells[i + 1][j] == BROKEN_HWALL : # UP
                    success_el.append(DOWN)

                if not j == 1 and cells[i][j - 1] == BROKEN_VWALL : # UP
                    success_el.append(LEFT)

                if not j == len(cells[0]) - 2 and cells[i][j + 1] == BROKEN_VWALL : # UP
                    success_el.append(RIGHT)

                success_row.append(success_el)
            successors.append(success_row)

        return successors
    
    def successors_str(self) :
        string = ''
        for row in self.successors :
            for col in row :
                for action in col :
                    string += action

                string += f','
            string = string[:-1] + '\n'
        
        return string[:-1]
            
    def str(self, action_seq = None) :
        cells = self.cells.copy()
        if action_seq != None :
            i = 1
            j = int(len(self.cells[0]) / 2)
            cells[0][j] = TRAVELED # top
            # other top
            cells[i][j] = TRAVELED
            cells[len(self.cells) - 1][j] = TRAVELED # bottom
            action_seq = f'{action_seq}'
            for action in action_seq :
                if action == UP :
                    i -= 1
                    cells[i][j] = TRAVELED
                    i -= 1
                    cells[i][j] = TRAVELED
                elif action == DOWN :
                    i += 1
                    cells[i][j] = TRAVELED
                    i += 1
                    cells[i][j] = TRAVELED
                elif action == LEFT :
                    j -= 1
                    cells[i][j] = TRAVELED_VWALL
                    j -= 1
                    cells[i][j] = TRAVELED
                elif action == RIGHT :
                    j += 1
                    cells[i][j] = TRAVELED_VWALL
                    j += 1
                    cells[i][j] = TRAVELED
            
        maze_str = ''
        for row in cells :
            for column in row :
                maze_str += CHARS[column]
            maze_str += '\n'
        return maze_str[:-1]

    def uninformed_search(self, mode) :
        path = [[UNSEARCHED for j in range(len(self.successors[0]))] for i in range(len(self.successors))]

        start_i, start_j = self.start
        goal_i, goal_j = self.goal
        

        frontier = [(start_i, start_j)]
        while len(frontier) > 0 :
            if mode == BFS :
                i, j = frontier.pop(0)
            elif mode == DFS :
                i, j = frontier.pop()
            else :
                raise(ValueError('bruh 3'))
            
            # visit current
            if path[i][j] == SEARCHED :
                continue

            path[i][j] = SEARCHED

            # end case
            if i == goal_i and j == goal_j :
                return path

            # children
            children = self.successors[i][j]
            #print(children, i, j)
            if mode == DFS :
                children = children.copy()
                children.reverse()

            #print(children, i, j)
            for action in children :
                next_state = None
                if action == UP :
                    next_state = (i - 1, j)
                elif action == DOWN :
                    next_state = (i + 1, j)
                elif action == LEFT :
                    next_state = (i, j - 1)
                elif action == RIGHT :
                    next_state = (i, j + 1)
                else :
                    print(f'bruh {action}')
                
                next_i, next_j = next_state

                if path[next_i][next_j] == UNSEARCHED :
                    frontier.append(next_state)

        raise(RuntimeError('Goal Not Found'))
    
    def path_str(path) :
        string = ''
        for row in path :
            for col in row :
                string += f'{col},'
            string = string[:-1] + '\n'
        return string[:-1]
    
    def distance(v1, v2, mode = EUCLIDEAN) :
        if len(v1) != len(v2) :
            raise(ValueError('incompatible vectors for distance'))
        
        sum = 0
        for i in range(len(v1)) :
            if mode == MANHATTEN :
                sum += abs(v1[i] - v2[i])
            elif mode == EUCLIDEAN :
                sum += (v1[i] - v2[i]) ** 2

        if mode == EUCLIDEAN :
            sum = sqrt(sum)
        
        return sum
    
    def distance_m(self, mode) :
        goal_i, goal_j = self.goal
        goal = [goal_i, goal_j]

        distance_m = []
        for i in range(len(self.successors)) :
            dist_row = []
            for j in range(len(self.successors[i])) :
                dist_row.append(RandomMaze.distance([i,j], goal, mode))
            distance_m.append(dist_row)
        
        return distance_m
    
    def A_star(self, mode) :
        path = [[UNSEARCHED for j in range(len(self.successors[0]))] for i in range(len(self.successors))]

        start_i, start_j = self.start
        start = [start_i, start_j]
        goal_i, goal_j = self.goal
        goal = [goal_i, goal_j]
        

        frontier = PriorityQueueAStar()
        frontier.enqueue((start_i, start_j, 0, RandomMaze.distance([start_i, start_j], goal, mode)))

        while len(frontier) > 0 :
            i, j, g, h = frontier.dequeue()
            
            # visit current
            if path[i][j] == SEARCHED :
                continue

            path[i][j] = SEARCHED

            # end case
            if i == goal_i and j == goal_j :
                return path

            # children
            children = self.successors[i][j]

            for action in children :
                next_i = None
                next_j = None
                if action == UP :
                    next_i = i - 1
                    next_j = j
                elif action == DOWN :
                    next_i = i + 1
                    next_j = j
                elif action == LEFT :
                    next_i = i 
                    next_j = j - 1
                elif action == RIGHT :
                    next_i = i
                    next_j = j + 1
                else :
                    print(f'bruh {action}')

                if path[next_i][next_j] == SEARCHED :
                    continue
                
                h = RandomMaze.distance([next_i, next_j], goal, mode)

                next_state = (next_i, next_j, g + 1, h)
                frontier.enqueue(next_state)

        raise(RuntimeError('Goal Not Found'))

    def action_sequence(self) :
        path = [[UNSEARCHED for j in range(len(self.successors[0]))] for i in range(len(self.successors))]

        start_i, start_j = self.start
        goal_i, goal_j = self.goal

        frontier = [(start_i, start_j, None)]
        action_seq = []
        while len(frontier) > 0 :
            i, j, prev_act = frontier.pop()
           
            # visit current
            #print(action_seq, prev_act)
            #print(frontier)
            if path[i][j] == SEARCHED :
                action_seq.pop()
                continue

            path[i][j] = SEARCHED
            action_seq.append(prev_act)

            # end case
            if i == goal_i and j == goal_j :
                return action_seq

            # children
            children = self.successors[i][j].copy()
            #print(children, i, j)
            children.reverse()

            # add current state to frontier
            frontier.append((i,j, prev_act))

            #print(children, i, j)
            for action in children :
                next_state = None
                if action == UP :
                    next_state = (i - 1, j)
                elif action == DOWN :
                    next_state = (i + 1, j)
                elif action == LEFT :
                    next_state = (i, j - 1)
                elif action == RIGHT :
                    next_state = (i, j + 1)
                else :
                    print(f'bruh {action}')
                
                next_i, next_j = next_state

                if path[next_i][next_j] == UNSEARCHED :
                    frontier.append((next_i, next_j, action))

        raise(RuntimeError('Goal Not Found'))

    def solved_maze(self, action_seq = None) :
        if action_seq == None :
            action_seq = self.action_sequence()
        
        maze_str = str(self)

        # get index of start and add in searched
        curr_i = -1
        curr_j = int(len(self.cells[0]) / 2)
        action_seq = f'{action_seq}D'

        # entry trace
        #str_ind = curr_i * (len(self.cells) + 1)  + curr_j
        #first_half = maze_str[:str_ind]
        #second_half = maze_str[str_ind + 1:]
        #maze_str = f'{first_half}{CHARS[TRAVELED]}{second_half}'

        # exit trace
        #str_ind = len(self.) * (len(self.cells) + 1)  + curr_j
        #first_half = maze_str[:str_ind]
        #second_half = maze_str[str_ind + 1:]
        #maze_str = f'{first_half}{CHARS[TRAVELED]}{second_half}'
        
        for move in action_seq :
            # update position
            if move == UP :
                curr_i -= 2
            elif move == DOWN :
                curr_i += 2
            elif move == LEFT :
                curr_j -= 2
            elif move == RIGHT :
                curr_j += 2

            # get string index
            str_ind = curr_i * (len(self.cells[0]) + 1)  + curr_j
            first_half = maze_str[:str_ind]
            second_half = maze_str[str_ind + 1:]
            maze_str = f'{first_half}{CHARS[TRAVELED]}{second_half}'
            
        return maze_str
        

class PriorityQueueAStar :
    """Assume elements in form of (i,j,g,h)"""
    def __init__ (self) :
        self.queue = []

    def enqueue(self, element) :
        i, j, g, h = element
        distance = g + h
        for k in range(len(self.queue)) :
            a,b,g2,h2 = self.queue[k]
            el_dis = g2 + h2
            if distance > el_dis :
                self.queue.insert(k, element)
                return
        self.queue.append(element)
    
    def dequeue(self) :
        return self.queue.pop()
    
    def __len__(self) :
        return len(self.queue)

# problem methods
def main(maze) :
    output = open(OUPUT_FILENAME, 'w')

    # q1
    output.write(f'P1Q1:\n\n{maze.str()}\n\n')

    # q2
    output.write(f'P1Q2:\n\n{maze.successors_str()}\n\n')

    # q3
    action_str = ''
    for i in maze.action_sequence() :
        if i == None :
            continue
        action_str += i
    output.write(f'P1Q3:\n\n{action_str}\n\n')
    
    # q4
    output.write(f'P1Q4:\n\n{maze.str(action_str)}\n\n')

    # q5
    path = maze.uninformed_search(BFS)
    output.write(f'P1Q5:\n\n{RandomMaze.path_str(path)}\n\n')

    # q6
    path = maze.uninformed_search(DFS)
    output.write(f'P1Q6:\n\n{RandomMaze.path_str(path)}\n\n')

    # q7
    
    output.write(f'P2Q7:\n\n{RandomMaze.path_str(maze.distance_m(MANHATTEN))}\n\n')

    # q8 
    path = maze.A_star(MANHATTEN)
    output.write(f'P2Q8:\n\n{RandomMaze.path_str(path)}\n\n')

    # q9
    path = maze.A_star(EUCLIDEAN)
    output.write(f'P2Q9:\n\n{RandomMaze.path_str(path)}\n\n')
    



# method calls
rand_maze = RandomMaze(HEIGHT, WIDTH)

main(rand_maze)