import numpy as np
import matplotlib.pyplot as plt
import copy

BSIZE = 3
EMPTY = 0
X = 1
O = 2
MYSIDE = X

XWIN = "XWIN"
OWIN = "OWIN"
DRAW = "DRAW"
NONTERMINAL = "NONTERMINAL"

# LAYERS = [64, 64, 1]
LAYERS = [64,32, 1]

DEBUG = False

def createBoard():
    board = np.empty([BSIZE,BSIZE],dtype=np.int32)
    board.fill(EMPTY)
    return board

    
def randArray(shape, scale):
    dd = np.prod(shape)
    rv = np.random.rand(dd)*2.0*scale-scale
    return rv.reshape(shape)
    
def randWeightsAndBias(in_dim, out_dim, scale):
    return [randArray([out_dim,in_dim,scale]),randArray([out_dim,1,scale])]
        
    
def createWeights(scale=1.0):
    rv =[]
    previous_layer = boardRep(createBoard(),X).shape[0]
    for new_layer in LAYERS:
        rv.append(randWeightsAndBias(previous_layer, new_layer, scale))
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
        val, state, rv = calcVal(board, move)
        print(val)
        side = {O:X,X:O}[side]
    print(board)
    print(state)
    return board, move
    
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
            
 
def learnXO(alpha = 0.001, epsilon = 0.05, num_games = 10000, 
            reset_weights = False, save_games=False, 
            learn_weights = True):
    global ALPHA
    ALPHA = alpha
    if 'weights' not in globals() or reset_weights:
        global weights
        weights = createWeights()
    wins = {OWIN:0,XWIN:0,DRAW:0}
    largest_cost_update = 0

    for ii in range(num_games):
        # orig_weights = weights.copy()
        board = createBoard()
        board_history = []
        state = NONTERMINAL
        current_side = X
        while state is NONTERMINAL:
            current_move, current_val, state = chooseNextMove(board, current_side, epsilon)
            makeMove(board, current_move)
            # print(board, current_move, current_val, state)
            tmp=(board.copy(), current_move, current_val, state)
            board_history.append(tmp)
            current_side = {O:X,X:O}[current_side]
    
        if DEBUG:
            print()
            print("NEW GAME:")
        
        b,m,v,s = board_history[-1]
        if s is DRAW:
            target_val = 0.5
        else:
            target_val = 1.0
        wins[s]+=1
        
        if save_games:
            targets=[target_val if i%2==0 else 1-target_val for i in range(len(board_history))]
            # targets=np.hstack(targets).reshape([1,-1])
            board_reps=[boardRep(b,m.side) for b,m,v,s in board_history]
            # board_reps=np.hstack(board_reps)
            
            global games_history
            if 'games_history' not in globals():
                games_history = [[board_reps], [targets]]
            else:
                games_history[0].append(board_reps)
                games_history[1].append(targets)
                
            if ii%100==0:
                print(wins)
        
        if learn_weights:
            for b,m,v,s in reversed(board_history):
                if DEBUG:
                    print("1")
                    print(b,m)
                val, state, deriv = calcValAndDeriv(b, m, target_val)
                max_dw, max_db = updateWeights(weights, deriv, alpha)
                new_val, junk1, junk2 = calcVal(b, m)
                if DEBUG:
                    print("val:",v,"target:",target_val,"val_after_update",new_val)
                cost = (new_val - val)**2
                if cost > largest_cost_update:
                    largest_cost_update=cost
                target_val = invertVal(new_val,state)
            if (ii+1)%100==0:
                print("num games:",ii+1,"cost:",largest_cost_update, 
                "wins:",wins)
                wins = {OWIN:0,XWIN:0,DRAW:0}
                largest_cost_update = 0
 
def updateWeights(weights, deriv, alpha):
    max_dw, max_db = np.float64('-inf'),np.float64('-inf') 
    for (w,b),(dw,db) in zip(weights, deriv):
        m_dw = dw.max()
        m_db = db.max()
        if m_dw>max_dw:
            max_dw = m_dw
        if m_db>max_db:
            max_db = m_db
        w-=alpha*dw
        b-=alpha*db
    return max_dw, max_db
            
def actionReward(board, move):
    makeMove(board, move)
    val, state, rv = calcVal(board, move)
    undoMove(board, move)
    return (val, state)

def makeMove(board, move):
    assert(board[move.ii,move.jj]==EMPTY)
    board[move.ii,move.jj]=move.side
    
def undoMove(board, move):
    assert(board[move.ii,move.jj]!=EMPTY)
    board[move.ii,move.jj]=EMPTY
    return board
    

def boardRep(board, side):
    other_side = {O:X,X:O}[side]
    tmp = board.reshape([-1,1])
    return np.vstack((tmp==side, tmp==other_side, tmp==EMPTY)).astype(np.float64)

def calcVal(board, last_move):
    win = checkWin(board, last_move)
    state = NONTERMINAL
    if win:
        state={O:OWIN,X:XWIN}[last_move.side]
        # val = 1.0
    elif (board==EMPTY).sum()==0:
        state = DRAW
        # val = 0.5
    board_rep = boardRep(board, last_move.side)
    rv = valFunc(board_rep, weights)
    val = rv[0][-1]
    return (val, state, rv)
    
def calcValAndDeriv(board, last_move, y):
    # forward
    final_val, state, (activations, zs) = calcVal(board, last_move)
    # backprop
    deriv = valFunc_deriv(activations, zs, weights, y)
    return (final_val, state, deriv)
    
def calcValAndDerivRaw(board_reps, y, weights):
    # forward
    (activations, zs) = nnFunc(board_reps, weights)
    # backprop
    deriv = valFunc_deriv(activations, zs, weights, y)
    return (activations, zs, deriv)

def fastLearn(fast_weights,
              n_cycles = 10,
              n_games = 10000, epsilon = 0.1, 
              n_updates = 1000, alpha = 1):
    for n in range(n_cycles):
        print("cycle:",n)
        global games_history
        if 'games_history' in globals():
            del games_history
        global weights
        weights = copy.deepcopy(fast_weights)
        learnXO(reset_weights=False, num_games=n_games, 
                alpha = 0.1, # not in use since learn_weights is False
                epsilon=epsilon, save_games=True,
                learn_weights=False)
        
#        rr = np.hstack(games_history[0])
#        tt = np.hstack(games_history[1])
#
#        learn_i(inp=rr, target=tt, weights=fast_weights, 
#                alpha=alpha, n_updates=n_updates)

def createLearningData(nsteps):
    bb=[]
    tt=[]
    for nn in range(1, nsteps+1):
        tmp_bb = np.hstack([games_history[0][i][-nn] for i in range(len(games_history[0])) if len(games_history[0][i])>=nn])
        tmp_tt = np.hstack([games_history[1][i][-nn] for i in range(len(games_history[1])) if len(games_history[1][i])>=nn]).reshape([1,-1])
        if len(tmp_bb)>0 and len(tmp_tt)>0:
            bb.append(tmp_bb)
            tt.append(tmp_tt)
        else:
            assert(len(tmp_bb)==0 and len(tmp_tt)==0)
    bb=np.hstack(bb)
    tt=np.hstack(tt)
    return bb,tt
def learn_i(inp, target, weights, alpha, n_updates, batch_size=1000):
    for i in range(1,n_updates):
        jj = np.random.randint(0,inp.shape[1],batch_size)
        activations, zs, deriv = calcValAndDerivRaw(inp[:,jj],target[:,jj],weights)
        max_dw, max_db = updateWeights(weights, deriv, alpha)
        if i%100==0:
            activations, zs, deriv = calcValAndDerivRaw(inp,target,weights)
            ee = activations[-1]-target
            print("max=",ee.max(),
                  "min=",ee.min(), 
                  "mean_abs=",abs(ee).mean(),
                  "mean_sqr=",(ee*ee).mean(),
                  "max dw=",max_dw,
                  "max db=",max_db)

def alternate_learn(inp, target, inp_m1, weights, alpha, n_updates):
    for i in range(1,n_updates):
        # learn_i(inp, target, weights, alpha, 101)
        aa, zz = nnFunc(inp, weights)
        target_m1 = 1.0-aa[-1]
        learn_i(np.hstack([inp,inp_m1]), 
                np.hstack([target,target_m1]), 
                weights, alpha, 101)
        if i%10==0:
            aa, zz = nnFunc(inp, weights)
            ee = aa[-1]-target
            aa, zz = nnFunc(inp_m1, weights)
            ee2 = aa[-1]-target_m1
            print("SUMMARY:",
                  "max=",ee.max(),
                  "min=",ee.min(), 
                  "mean_sqr=",(ee*ee).mean(),
                  "max_m1=",ee2.max(),
                  "min_m1=",ee2.min(), 
                  "mean_sqr_m1=",(ee2*ee2).mean()  
                 )
    
def invertVal(val, state):
    if state is DRAW:
        return val
    return 1.0 - val             
          
## backpropagation algorithm          
def valFunc_deriv(activations, zs, weights, y):
    activation = activations[-1]
    z = zs[-1]
    nabla_a_c = costFunc_deriv(activation, y)
    deltas = [nabla_a_c * outFunc_deriv(z)]
    dC_dw = []
    dC_db = []
    for activation, z, (W, b) in zip(reversed(activations[1:-1]), 
                                     reversed(zs[:-1]), 
                                     reversed(weights[1:])):                             
        deltas.append(np.dot(W.T,deltas[-1]) * sigmaFunc_deriv(z))
    n_samples = deltas[-1].shape[1]
    ones = np.ones(dtype=np.float64, shape=[n_samples,1])
    for activation, delta in zip(activations[:-1], reversed(deltas)):
        dC_dw.append(np.dot(delta,activation.T)/n_samples) 
        dC_db.append(np.dot(delta, ones)/n_samples)
    dC = [(dw,db) for dw,db in zip(dC_dw,dC_db)]
    return dC
    
def outFunc(x):
    return 1.0/(1.0+np.exp(-x))

def outFunc_deriv(x):
    e_x = np.exp(-x)
    tmp = 1+e_x
    return e_x/(tmp*tmp)

## assuming cost function:  0.5*(activation - y)^2
# could try: -y*log(y_hat)-(1-y)*log(1-y_hat)
def costFunc_deriv(activation, y):
    return activation - y
    
def valFunc(x, weights):
    activations, zs = nnFunc(x,weights)
    return (activations, zs)

def nnFunc(x, weights):
    activation = x
    activations = [x]
    zs = []
    for W,b in weights[:-1]:
        z = np.dot(W,activation)+b
        activation = sigmaFunc(z)
        zs.append(z)
        activations.append(activation)
    W,b = weights[-1]
    z = np.dot(W,activation)+b
    activation = outFunc(z)
    zs.append(z)
    activations.append(activation)
    return (activations, zs)

def sigmaFunc(z):
    return z*(z>0)

def sigmaFunc_deriv(z):
    return 1.0*(z>0)

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


