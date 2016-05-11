# this is a test
import numpy as np
BSIZE = 4
EMPTY = 0
X = 1
O = 2
MYSIDE = X

XWIN = "XWIN"
OWIN = "OWIN"
DRAW = "DRAW"
NONTERMINAL = "NONTERMINAL"

LAYERS = [64, 64, 1]

DEBUG = False

def createBoard():
    board = np.empty([BSIZE,BSIZE],dtype=np.int32)
    board.fill(EMPTY)
    return board

    
def randArray(shape):
    dd = np.prod(shape)
    rv = np.random.rand(dd)*2.0-1.0
    return rv.reshape(shape)
    
def randWeightsAndBias(in_dim, out_dim):
    return [randArray([out_dim,in_dim]),randArray([out_dim,1])]
        
    
def createWeights():
    rv =[]
    previous_layer = boardRep(createBoard(),X).shape[0]
    for new_layer in LAYERS:
        rv.append(randWeightsAndBias(previous_layer, new_layer))
        previous_layer = new_layer
    return rv


class Move:
    def __init__(self, _ijk_tuple, _side):
        self.ii=_ijk_tuple[0]
        self.jj=_ijk_tuple[1]
        self.side=_side
    def __repr__(self):
        return "Move(%s,%s)" % ((self.ii, self.jj),["none","X","O"][self.side])

def chooseNextMove(board, side, epsilon):
    it = np.nditer(board, flags=['multi_index'])
    if epsilon>0 and np.random.rand() < epsilon:
        num_empty = (board==EMPTY).sum()
        nchoice = np.random.randint(0,num_empty)
        nn = 0
        while not it.finished:
            if it[0]==EMPTY:
                if nn==nchoice:
                    move = Move(it.multi_index,side)
                    val,state = actionReward(board, move)
                    return (move, val, state)
                nn+=1
            it.iternext()
    else:
        best_move = None
        best_val = float('-inf')
        best_state = NONTERMINAL
        while not it.finished:
            if it[0]==EMPTY:
                move = Move(it.multi_index,side)
                val,state = actionReward(board, move)
                if val>best_val or best_move is None:
                    best_val = val
                    best_move = move
                    best_state = state
            it.iternext()
    return (best_move, best_val, best_state)
 
def playHuman(human_first=True):
    board = createBoard()
    move = Move((0,0,0),O)
    state = NONTERMINAL
    side = X
    human_side = human_first and X or O
    while state is NONTERMINAL:
        print(board)
        if side==human_side:
            move = getMove(board, side)
        else:
            move, c_val, c_state = chooseNextMove(board, side, epsilon=0)
        makeMove(board, move)
        val, state = calcVal(board, move)
        print(val)
        side = {O:X,X:O}[side]
    print(board)
    print(state)
    
def getMove(board, side):
    good = False
    while not good:
        mm = input("your move: (enter x,y)\n")
        try:
            mm=eval(mm)
            assert( type(mm) is tuple and 
                    len(mm)==2 and 
                    all([type(i) is int and i>=1 and i<=BSIZE for i in mm]))
        except(NameError, AssertionError):
            print("bad input")
            continue
        mm = [i-1 for i in mm]
        if board[mm[0],mm[1]]!=EMPTY:
            print("place taken, choose another place")
            continue
        good = True
    move = Move(mm,side)
    return move
            
 
def learnXO(alpha = 0.001, epsilon = 0.05, num_games = 10000, reset_weights = False):
    global ALPHA
    ALPHA = alpha
    if 'weights' not in globals() or reset_weights:
        global weights
        weights = createWeights()
    wins = {OWIN:0,XWIN:0,DRAW:0}
    largest_val_update = 0
    for ii in range(num_games):
        # orig_weights = weights.copy()
        board = createBoard()
        board_history = []
        state = NONTERMINAL
        current_side = X
        while state is NONTERMINAL:
            current_move, current_val, state = chooseNextMove(board, current_side, epsilon)
            makeMove(board, current_move)
            tmp=(board.copy(), current_move, current_val, state)
            board_history.append(tmp)
            current_side = {O:X,X:O}[current_side]
        if DEBUG:
            print(board_history[-1])
        b,m,v,s = board_history[-1]
        next_val = invertVal(v, m.side)
        wins[s]+=1
        for b,m,v,s in reversed(board_history[:-1]):
            val, state = calcVal(b, m)
            val_diff = next_val - val
            deriv = calcDeriv(b, m.side)
            weights += ALPHA * val_diff * deriv
            # for debug
            val2, junk = calcVal(b, m)
            val2_diff = next_val - val2
            val_update = abs(val_diff)-abs(val2_diff)
            if val_update > largest_val_update:
                largest_val_update=val_update
            orig_next_val = next_val
            next_val = invertVal(val2,state)
            if DEBUG:
                print("orig_val:",val,"target:",orig_next_val,"new_val:",val2, "next_target:",next_val)
        if (ii+1)%100==0:
            print("num games:",ii+1,"largest val change:",largest_val_update, 
            "wins:",wins)
            wins = {OWIN:0,XWIN:0,DRAW:0}
            largest_val_update = 0
    
       
def actionReward(board, move):
    makeMove(board, move)
    val, state = calcVal(board, move)
    undoMove(board, move)
    return (val, state)

def makeMove(board, move):
    assert(board[move.ii,move.jj]==EMPTY)
    board[move.ii,move.jj]=move.side
    
def undoMove(board, move):
    assert(board[move.ii,move.jj]!=EMPTY)
    board[move.ii,move.jj]=EMPTY
    

def boardRep(board, side):
    other_side = {O:X,X:O}[side]
    tmp = board.reshape([-1,1])
    return np.vstack((tmp==side, tmp==other_side, tmp==EMPTY)).astype(np.float64)

def calcVal(board, last_move):
    win = checkWin(board, last_move)
    if win:
        final_state={O:OWIN,X:XWIN}[last_move.side]
        return (1.0,final_state)
    if (board==EMPTY).sum()==0:
        return (0.5, DRAW)
    board_rep = boardRep(board, last_move.side)
    rv = valFunc(board_rep, weights)
    # print(rv)
    val = rv[2][0][0]
    return (val, NONTERMINAL)
    
def invertVal(val, state):
    if state is DRAW:
        return val
    return 1.0 - val             
          
def calcDeriv(board, last_side):
    board_rep = boardRep(board, last_side)
    deriv = valFuncDeriv(board_rep, weights)
    return deriv

def outFunc(x):
    return 1.0/(1.0+np.exp(x))

def outFunc_deriv(x):
    e_x = np.exp(x)
    return -e_x*np.power(1+e_x,-2)
    

def valFunc(x, weights):
    (activations, zs) = nnFunc(x,weights)
    val = outFunc(activations[-1])
    return (activations, zs, val)

def nnFunc(x, weights):
    activation = x
    activations = [x]
    zs = []
    for W,b in weights:
        z = np.dot(W,activation)+b
        activation = sigmaFunc(z)
        zs.append(z)
        activations.append(activation)
    return (activations, zs)

def sigmaFunc(z):
    return z*(z>0)

def sigmaFunc_deriv(z):
    return 1.0*(z>0)

# use (y-y_hat)**2 OR 
# -y*log(y_hat)-(1-y)*log(1-y_hat)
def costFunc_deriv():
    None
    
def checkWin(board, last_move):
    val = last_move.side
    ii = last_move.ii
    jj = last_move.jj
    won = True
    for nn in range(0,BSIZE):
        if board[ii,nn]!=val:
            won = False
            break
    if won:
        return True
    won = True
    for nn in range(0,BSIZE):
        if board[nn,jj]!=val:
            won = False
            break
    if won:
        return True
    if ii==jj:
        won = True
        for nn in range(0,BSIZE):
            if board[nn,nn]!=val:
                won = False
                break
        if won:
            return True
    if ii==BSIZE-1-jj:
        won = True 
        for nn in range(0,BSIZE):
            if board[nn,BSIZE-1-nn]!=val:
                won = False
                break
        if won:
            return True
    return False
    


##########################################################

def update_mini_batch(self, mini_batch, eta):
    """Update the network's weights and biases by applying
    gradient descent using backpropagation to a single mini batch.
    The "mini_batch" is a list of tuples "(x, y)", and "eta"
    is the learning rate."""
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [w-(eta/len(mini_batch))*nw 
                    for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb 
                   for b, nb in zip(self.biases, nabla_b)]
                           
def backprop(self, x, y):
    """Return a tuple "(nabla_b, nabla_w)" representing the
    gradient for the cost function C_x.  "nabla_b" and
    "nabla_w" are layer-by-layer lists of numpy arrays, similar
    to "self.biases" and "self.weights"."""
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)
    # backward pass
    delta = self.cost_derivative(activations[-1], y) * \
        sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in range(2, self.num_layers):
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)


def cost_derivative(self, output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    return (output_activations-y) 

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#######################################################################
