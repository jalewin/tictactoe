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
LAYERS = [64, 32, 32, 1]

DEBUG = False

def createBoard():
    board = np.empty([BSIZE,BSIZE],dtype=np.int32)
    board.fill(EMPTY)
    return board

def createScoreBoard():
    board = np.empty([BSIZE,BSIZE],dtype=np.float64)
    board.fill(-1.0)
    return board
    
def randArray(shape, scale):
    dd = np.prod(shape)
    rv = np.random.rand(dd)*2.0*scale-scale
    return rv.reshape(shape)
    
def randWeightsAndBias(in_dim, out_dim, scale):
    return [randArray([out_dim,in_dim],scale),randArray([out_dim,1],scale)]
        
    
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
    scores = createScoreBoard()
    scores[board==EMPTY] = -2

    while not it.finished:
        if it[0]==EMPTY:
            move = Move(it.multi_index,side)
            val = actionReward(board, move)
            # print(it.multi_index, val)
            scores[it.multi_index] = val
        it.iternext()
    
    # returns first instance of max score
    best_idx = np.unravel_index(scores.argmax())
    best_val = scores[best_idx]
    best_move = Move(best_idx, side)
    chose_rand = False
    
    if epsilon>0 and np.random.rand() < epsilon:
        idxs = np.where(board==EMPTY)
        ii = np.random.randint(0,len(idxs[0]))
        rand_idx = tuple([idxs[j][ii] for j in range(len(idxs))])
        best_move = Move(rand_idx, side)
        chose_rand = True
    
    
    win = checkWin(board, best_move)
    state = NONTERMINAL
    if win:
        state={O:OWIN,X:XWIN}[best_move.side]
        # val = 1.0
    elif (board==EMPTY).sum()==0:
        state = DRAW
        # val = 0.5  

    return (best_move, best_val, state, chose_rand, scores)
    
 
def playHuman(human_first=True):
    board = createBoard()
    move = Move((0,0,0),O)
    state = NONTERMINAL
    side = X
    human_side = X if human_first else O
    while state is NONTERMINAL:
        print()
        print(board)
        if side==human_side:
            move = getMove(board, side)
        else:
            move, c_val, c_state, c_rand, scores = chooseNextMove(board, side, epsilon=0)
        
        makeMove(board, move)
        val, rv = calcVal(board, move)
        ## TODOJ - add checkWIn to calculate state
       
        if side==human_side:
            print("val=",val)
        else:
            print(scores)
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
            # TODOJ - choose only 1 random move in a game? need to make the random choice at a uniform stage
            current_move, current_val, state, is_rand, scores = chooseNextMove(board, current_side, epsilon)
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
            # TODOJ - make targets the value of the nnFunc and only the last value the game value.
            targets=[target_val if i%2==0 else 1-target_val for i in range(len(board_history))]
            targets.reverse()
            # targets=np.hstack(targets).reshape([1,-1])
            board_reps=[boardRep(b,m.side) for b,m,v,s in board_history]
            # board_reps=np.hstack(board_reps)
            
            global games_history
            if 'games_history' not in globals():
                games_history = [[board_reps], [targets]]
            else:
                games_history[0].append(board_reps)
                games_history[1].append(targets)
        
        if learn_weights:
            for b,m,v,s in reversed(board_history):
                if DEBUG:
                    print("1")
                    print(b,m)
                val, deriv = calcValAndDeriv(b, m, target_val)
                max_dw, max_db = updateWeights(weights, deriv, alpha)
                new_val, junk1 = calcVal(b, m)
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
    
    print(wins)
 
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
    val, rv = calcVal(board, move)
    undoMove(board, move)
    return val

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
    # return np.vstack((tmp==side, tmp==other_side, tmp==EMPTY)).astype(np.float64)
    return np.vstack((side==X, tmp==X, tmp==O, tmp==EMPTY)).astype(np.float64)

def calcVal(board, last_move):
#    win = checkWin(board, last_move)
#    state = NONTERMINAL
#    if win:
#        state={O:OWIN,X:XWIN}[last_move.side]
#        # val = 1.0
#    elif (board==EMPTY).sum()==0:
#        state = DRAW
#        # val = 0.5
    board_rep = boardRep(board, last_move.side)
    rv = valFunc(board_rep, weights)
    val = rv[0][-1]
    return (val, rv)
    #  return (val, state, rv)
    
def calcValAndDeriv(board, last_move, y):
    # forward
    final_val, (activations, zs) = calcVal(board, last_move)
    # backprop
    deriv = valFunc_deriv(activations, zs, weights, y)
    return (final_val, deriv)
    
def calcValAndDerivRaw(board_reps, y, weights):
    # forward
    (activations, zs) = nnFunc(board_reps, weights)
    # backprop
    deriv = valFunc_deriv(activations, zs, weights, y)
    return (activations, zs, deriv)

def fastLearn(fast_weights,
              n_cycles = 10,
              n_games = 10000, epsilon = 0.1, 
              n_processed = 1e7, decay = 0.7, alpha = 1, batch_size=0):
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
                
        dd=createLearningData(games_history)
        
        max_processed = n_processed
        for ii in range(len(dd)):
            print("stage",ii, "data set size",dd[ii][0].shape)
            learn_i(inp=dd[ii][0], target=dd[ii][1], weights=fast_weights, 
                    alpha=alpha, max_processed=max_processed, batch_size=batch_size)
            max_processed *= decay
        

def createLearningData(games):
    input_target_list=[]
    nn = 0
    while True:
        nn += 1
        tmp_bb = [games[0][i][-nn] for i in range(len(games[0])) 
                  if len(games[0][i])>=nn]
        tmp_tt = [games[1][i][-nn] for i in range(len(games[1])) 
                  if len(games[1][i])>=nn]
        if len(tmp_bb)>0 and len(tmp_tt)>0:
            input_target_list.append([np.hstack(tmp_bb), 
                                      np.hstack(tmp_tt).reshape([1,-1])])
        else:
            assert(len(tmp_bb)==0 and len(tmp_tt)==0)
            break 
    return input_target_list
    
    
def learn_i(inp, target, weights, alpha, max_processed, batch_size=1000):
    processed = 0
    next_print = 0
    while processed < max_processed:
        if batch_size > 0:
            jj = np.random.randint(0,inp.shape[1],batch_size)
            activations, zs, deriv = calcValAndDerivRaw(inp[:,jj],target[:,jj],weights)
            processed += batch_size
        else:
            activations, zs, deriv = calcValAndDerivRaw(inp, target, weights)
            processed += inp.shape[1]
        max_dw, max_db = updateWeights(weights, deriv, alpha)
        if processed > next_print:
            next_print = processed + 1e6
            activations, zs, deriv = calcValAndDerivRaw(inp,target,weights)
            ee = activations[-1]-target
            print("max=",ee.max(),
                  "min=",ee.min(), 
                  "mean_abs=",abs(ee).mean(),
                  "mean_sqr=",(ee*ee).mean(),
                  "max dw=",max_dw,
                  "max db=",max_db)

    
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
    
def checkWin(board, move):
    board = makeMove(board, move)
    rv = checkWin_i(board, move)
    board = undoMove(board, move)
    return rv

def checkWin_i(board, last_move):
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
    
