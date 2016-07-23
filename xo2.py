import numpy as np
# from ggplot import *
import matplotlib.pyplot as plt
import copy
import pandas as pd

LAYERS = [64, 64, 1]
# LAYERS = [64, 32, 32, 1]




DEBUG = False


class Move:
    def __init__(self, _index_tuple, _side):
        self.idx=_index_tuple
        self.side=_side
    def __repr__(self):
        return "Move(%s,%s)" % ((self.idx),["none","X","O"][self.side])
        
def dummyFun1(x,y):
    return [x for i in range(y)]
    
class Board:
    DIM = 2
    SIZE = 3
    EMPTY = 0
    X = 1
    O = 2
    XWIN = "XWIN"
    OWIN = "OWIN"
    DRAW = "DRAW"
    NONTERMINAL = "NONTERMINAL"
    _nvals = np.power(SIZE, DIM)
    # need dummy fun in python3 since comprehension expression cannot use class attributes
    # dummyFun1(SIZE, DIM)
    _shape = [SIZE]*DIM  
    side_to_string = [" ","X","O"]
    
    def __init__(self, board_repr=None):
        self.move_hist = []
        self.last_side = Board.O
        self.board = np.empty(Board._shape,dtype=np.int32)
        self.board.fill(Board.EMPTY)
        if board_repr is not None:
            self.board[(board_repr[1:(1+Board._nvals):1]==1).reshape(Board._shape)]=Board.X
            self.board[(board_repr[(1+Board._nvals):(1+2*Board._nvals):1]==1).reshape(Board._shape)]=Board.O
            self.last_side = Board.X if board_repr[0]==1 else Board.O
            
    def __repr__(self):
        rv="\nlast_side=%s" % Board.side_to_string[self.last_side] + "\n"
        tmp = self.board.__repr__().replace("array([","").replace("])","").replace(
            ",\n       ","\n").replace(",","").replace(str(Board.X),"X").replace(str(Board.O),"O").replace(str(Board.EMPTY)," ")
        rv+=tmp + "\n"
        return rv
        
    def otherSide(side):
        return {Board.X:Board.O,Board.O:Board.X}[side]
        
    def nextSide(self):
        return Board.otherSide(self.last_side)

    def boardRep(self):
        # other_side = {Board.O:Board.X,Board.X:Board.O}[side]
        tmp = self.board.reshape([-1,1])
        # return np.vstack((tmp==side, tmp==other_side, tmp==Board.EMPTY)).astype(np.float64)
        return np.vstack((self.last_side==Board.X, tmp==Board.X, tmp==Board.O, tmp==Board.EMPTY)).astype(np.float64)         
    
    def inputMove(self):
        good = False
        while not good:
            mm = input("your move: (enter x,y)\n")
            try:
                mm = eval(mm)
                assert( type(mm) is tuple and 
                        len(mm)==Board.DIM and 
                        all([type(i) is int and i>=1 and i<=Board.SIZE for i in mm]))
            except(NameError, AssertionError):
                print("bad input")
                continue
            mm = [i-1 for i in mm]
            # if self.board[mm[0],mm[1]]!=Board.EMPTY:
            if self.board[mm]!=Board.EMPTY:
                print("place taken, choose another place")
                continue
            good = True
        move = Move(mm, self.nextSide())
        return move
            
        
    def makeMove(self, move):
        if self.board[move.idx]!=Board.EMPTY or self.last_side == move.side:
            print(self.board)
            print(self.last_side)
            print(move)
            assert(self.board[move.idx]==Board.EMPTY)
            assert(self.last_side != move.side)
        self.board[move.idx]=move.side
        self.last_side = move.side
        self.move_hist.append(move)
    
    def undoMove(self):
        assert(len(self.move_hist) > 0)
        move = self.move_hist.pop()
        if self.board[move.idx]==Board.EMPTY or self.last_side != move.side:
            print(self.board)
            print(self.last_side)
            print(move)
            assert(self.board[move.idx]!=Board.EMPTY)
            assert(self.last_side == move.side)
        self.board[move.idx]=Board.EMPTY
        self.last_side = Board.otherSide(move.side)
                 
    def checkWin(self, move):
        self.makeMove(move)
        win = self.checkWin_i()
        state = Board.NONTERMINAL
        if win:
            state={Board.O:Board.OWIN,Board.X:Board.XWIN}[move.side]
        elif (self.board==Board.EMPTY).sum()==0:
            state = Board.DRAW
        self.undoMove()
        return state
    
    def checkWin_i(self):
        if len(self.move_hist)>0:
            last_idx = self.move_hist[-1].idx
        else:
            last_idx = None
        assert(last_idx is not None) # TODO If None, check all possibilities
        if Board.DIM==2:
            return self.checkWin_2d(last_idx)
        elif Board.DIM==3:
            return self.checkWin_3d(last_idx)
        else:
            print("only 2d or 3d supported")
            assert(Board.DIM==2 or Board.DIM==3)
    
    def checkWin_2d(self, last_idx):
        val = self.last_side
        won = True
        for nn in range(0,Board.SIZE):
            tmp = last_idx[:-1] + (nn,)
            if self.board[tmp]!=val:
                won = False
                break
        if won:
            return True
        won = True
        for nn in range(0,Board.SIZE):
            tmp = (nn,) + last_idx[1:]
            if self.board[tmp]!=val:
                won = False
                break
        if won:
            return True
        if last_idx[0]==last_idx[1]:
            won = True
            for nn in range(0,Board.SIZE):
                tmp = (nn,nn)
                if self.board[tmp]!=val:
                    won = False
                    break
            if won:
                return True
        if last_idx[0]==Board.SIZE-1-last_idx[1]:
            won = True 
            for nn in range(0,Board.SIZE):
                tmp = (nn, Board.SIZE-1-nn)
                if self.board[tmp]!=val:
                    won = False
                    break
            if won:
                return True
        return False
    
    def checkWin_3d(self, last_idx):
        val = self.last_side        
        won = True
        for nn in range(0,Board.SIZE):
            tmp = last_idx[:-1] + (nn,)
            if self.board[tmp]!=val:
                won = False
                break
        if won:
            return True
        won = True
        for nn in range(0,Board.SIZE):
            tmp = last_idx[0] + (nn,) + last_idx[2]
            if self.board[tmp]!=val:
                won = False
                break
        if won:
            return True
        won = True
        for nn in range(0,Board.SIZE):
            tmp = (nn,) + last_idx[1] + last_idx[2] 
            if self.board[tmp]!=val:
                won = False
                break
        if won:
            return True
        if last_idx[0]==last_idx[1]:
            won = True
            for nn in range(0,Board.SIZE):
                tmp = (nn,nn) + last_idx[2]
                if self.board[tmp]!=val:
                    won = False
                    break
            if won:
                return True
        if last_idx[0]==Board.SIZE-1-last_idx[1]:
            won = True 
            for nn in range(0,Board.SIZE):
                tmp = (nn, Board.SIZE-1-nn) + last_idx[2]
                if self.board[tmp]!=val:
                    won = False
                    break
            if won:
                return True
        if last_idx[0]==last_idx[2]:
            won = True
            for nn in range(0,Board.SIZE):
                tmp = (nn,) + last_idx[1] + (nn,)
                if self.board[tmp]!=val:
                    won = False
                    break
            if won:
                return True
        if last_idx[0]==Board.SIZE-1-last_idx[2]:
            won = True 
            for nn in range(0,Board.SIZE):
                tmp = (nn,) + last_idx[1] + (Board.SIZE-1-nn,)
                if self.board[tmp]!=val:
                    won = False
                    break
            if won:
                return True
        if last_idx[1]==last_idx[2]:
            won = True
            for nn in range(0,Board.SIZE):
                tmp = tmp[0] + (nn,nn)
                if self.board[tmp]!=val:
                    won = False
                    break
            if won:
                return True
        if last_idx[1]==Board.SIZE-1-last_idx[2]:
            won = True 
            for nn in range(0,Board.SIZE):
                tmp = tmp[0] + (nn,Board.SIZE-1-nn)
                if self.board[tmp]!=val:
                    won = False
                    break
            if won:
                return True
        if last_idx[0]==last_idx[1] and last_idx[1]==last_idx[2]:
            won = True
            for nn in range(0,Board.SIZE):
                tmp = (nn,nn,nn)
                if self.board[tmp]!=val:
                    won = False
                    break
            if won:
                return True
        if last_idx[0]==last_idx[1] and last_idx[1]==Board.SIZE-1-last_idx[2]:
            won = True
            for nn in range(0,Board.SIZE):
                tmp = (nn,nn,Board.SIZE-1-nn)
                if self.board[tmp]!=val:
                    won = False
                    break
            if won:
                return True
        if last_idx[0]==Board.SIZE-1-last_idx[1] and last_idx[1]==last_idx[2]:
            won = True
            for nn in range(0,Board.SIZE):
                tmp = (nn,Board.SIZE-1-nn,nn)
                if self.board[tmp]!=val:
                    won = False
                    break
            if won:
                return True
        if last_idx[0]==Board.SIZE-1-last_idx[1] and last_idx[1]==Board.SIZE-1-last_idx[2]:
            won = True
            for nn in range(0,Board.SIZE):
                tmp = (nn,Board.SIZE-1-nn,Board.SIZE-1-nn)
                if self.board[tmp]!=val:
                    won = False
                    break
            if won:
                return True
        return False
    
                  
class VanillaOptim:
    def __init__(self, max_processed = 1e8, batch_size = 1000, alpha=0.1):
        self.alpha=alpha
        self.max_processed=max_processed
        self.batch_size=batch_size
        print((self.alpha,self.max_processed,self.batch_size))
        
    def update(self, weights, deriv):
        max_dw, max_db = np.float64('-inf'),np.float64('-inf') 
        for (w,b),(dw,db) in zip(weights, deriv):
            m_dw = dw.max()
            m_db = db.max()
            if m_dw>max_dw:
                max_dw = m_dw
            if m_db>max_db:
                max_db = m_db
            w-=self.alpha*dw
            b-=self.alpha*db
        return max_dw, max_db
    
    def optimize(self, inp, target, weights):
         processed = 0
         next_print = 0
         while processed < self.max_processed:
            if self.batch_size > 0:
                jj = np.random.randint(0,inp.shape[1],self.batch_size)
                activations, zs, deriv = calcValAndDerivRaw(inp[:,jj],target[:,jj],weights)
                processed += self.batch_size
            else:
                activations, zs, deriv = calcValAndDerivRaw(inp, target, weights)
                processed += inp.shape[1]
                
            max_dw, max_db = self.update(weights, deriv)
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
    

class AdamOptim:
    def __init__(self, max_processed = 1e8, batch_size = 0, 
                 alpha=0.001, b1=0.9, b2=0.999, eps=1e-8,
                 bAdaMax = False):
        self.max_processed = max_processed
        self.batch_size = batch_size
        self.alpha=alpha
        self.b1=b1
        self.b2=b2
        self.eps=eps
        self.m_v=None
        self.t=0
        self.updateFun = self.updateAdaMax if bAdaMax else self.updateAdam
        assert(b1>=0 and b1<1)
        assert(b2>=0 and b2<1)
        
    def initialize_m_v(self, deriv):
        self.m_v = [[[np.zeros_like(dw), np.zeros_like(db)], 
                     [np.zeros_like(dw), np.zeros_like(db)]] for dw,db in deriv]
                    
    def updateAdam(self, weights, deriv):
        for (((m1,m2),(v1,v2)), (dw, db), (w,b)) in zip(self.m_v, deriv, weights):
            m1 = self.b1*m1 + (1-self.b1)*dw
            m2 = self.b1*m2 + (1-self.b1)*db
            v1 = self.b2*v1 + (1-self.b2)*np.square(dw)
            v2 = self.b2*v2 + (1-self.b2)*np.square(db)
            b1_t = np.power(self.b1, self.t)
            b2_t = np.power(self.b2, self.t)
            m1_hat = m1 / (1-b1_t)
            m2_hat = m2 / (1-b1_t)
            sqrt_v1_hat = np.sqrt(v1 / (1-b2_t)) + self.eps
            sqrt_v2_hat = np.sqrt(v2 / (1-b2_t)) + self.eps
            w -= self.alpha*m1_hat/sqrt_v1_hat
            b -= self.alpha*m2_hat/sqrt_v2_hat
    
    def updateAdaMax(self, weights, deriv):
        for (((m1,m2),(v1,v2)), (dw, db), (w,b)) in zip(self.m_v, deriv, weights):
            m1 = self.b1*m1 + (1-self.b1)*dw
            m2 = self.b1*m2 + (1-self.b1)*db
            v1 = np.maximum(self.b2*v1, np.abs(dw))
            v2 = np.maximum(self.b2*v2, np.abs(db))
            b1_t = np.power(self.b1, self.t)
            m1_hat = m1 / (1-b1_t)
            m2_hat = m2 / (1-b1_t)
            v1_hat = v1 + self.eps
            v2_hat = v2 + self.eps
            w -= self.alpha*m1_hat/v1_hat
            b -= self.alpha*m2_hat/v2_hat
            
    
    def optimize(self, inp, target, weights):
        processed = 0
        next_print = 0
        while processed < self.max_processed:
            self.t+=1
            if self.batch_size > 0:
                jj = np.random.randint(0,inp.shape[1],self.batch_size)
                activations, zs, deriv = calcValAndDerivRaw(inp[:,jj],target[:,jj],weights)
                processed += self.batch_size
            else:
                activations, zs, deriv = calcValAndDerivRaw(inp, target, weights)
                processed += inp.shape[1]
            if self.m_v is None:
                self.initialize_m_v(deriv)
            self.updateFun(weights, deriv)
            if processed > next_print:
                next_print = processed + 1e6
                activations, zs, deriv = calcValAndDerivRaw(inp,target,weights)
                ee = activations[-1]-target
                print("max=",ee.max(),
                      "min=",ee.min(), 
                      "mean_abs=",abs(ee).mean(),
                      "mean_sqr=",(ee*ee).mean())
                      

def createWinningBoards():
    brds={Board.X:[],Board.O:[]}
    for side in [Board.X,Board.O]:
        for j in range(Board.SIZE):
            bb = Board() # createBoard()
            for i in range(Board.SIZE): 
                bb.makeMove(Move((j,i),side))
                bb2 = bb.copy()
            brds[side].append(bb.board)
            brds[side].append(bb2.board.T)
        bb = Board() # createBoard()
        for i in range(Board.SIZE):
            bb.makeMove(Move((i,i), side))
        brds[side].append(bb.board)
        bb = Board() # createBoard()
        for i in range(Board.SIZE):
            bb.makeMove(Move((i,Board.SIZE-1-i), side))
        brds[side].append(bb.board)
    return brds

def createScoreBoard():
    board = np.empty([Board.SIZE for d in range(Board.DIM)],dtype=np.float64)
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
    previous_layer = Board().boardRep().shape[0]
    for new_layer in LAYERS:
        rv.append(randWeightsAndBias(previous_layer, new_layer, scale))
        previous_layer = new_layer
    return rv

def getScores(weights, board):
    it = np.nditer(board.board, flags=['multi_index'])
    scores = createScoreBoard()
    scores[board.board==Board.EMPTY] = -2
    next_side = board.nextSide()
    while not it.finished:
        if it[0]==Board.EMPTY:
            move = Move(it.multi_index,next_side)
            val = actionReward(weights, board, move)
            # print(it.multi_index, val)
            scores[it.multi_index] = val
        it.iternext()   
    return scores

def chooseNextMove(weights, board, epsilon):
    scores = getScores(weights, board)
 
    # returns first instance of max score
    best_idx = np.unravel_index(scores.argmax(), scores.shape)
    best_val = scores[best_idx]
    best_move = Move(best_idx, board.nextSide())
    chose_rand = False
    
    if epsilon>0 and np.random.rand() < epsilon:
        idxs = np.where(board==Board.EMPTY)
        if len(idxs[0])>0:
            ii = np.random.randint(0,len(idxs[0]))
            rand_idx = tuple([idxs[j][ii] for j in range(len(idxs))])
            best_move = Move(rand_idx, board.nextSide())
            chose_rand = True
     
    # note that val is the max value. In case of random move might not be val of move
    return (best_move, best_val, chose_rand, scores)
    
 
def playHuman(weights, human_first=True):
    board = Board() 
    state = Board.NONTERMINAL
    human_side = Board.X if human_first else Board.O
    while state is Board.NONTERMINAL:
        print()
        print(board)
        if board.nextSide()==human_side:
            move = board.inputMove()
        else:
            move, c_val, c_rand, scores = chooseNextMove(weights, board, epsilon=0)
        
        state = board.checkWin(move)
        board.makeMove(move)
        val, rv = calcVal(weights, board.boardRep())
       
        if board.last_side==human_side:
            print("val=",val)
        else:
            print(scores)
    print(board)
    print(state)
    return board, move
 
def learnXO(weights, alpha = 0.001, epsilon = 0.05, num_games = 10000, 
            reset_weights = False, save_games=False, 
            learn_weights = True):
    global ALPHA
    ALPHA = alpha
    wins = {Board.OWIN:0,Board.XWIN:0,Board.DRAW:0}
    largest_cost_update = 0

    for ii in range(num_games):
        board = Board()
        board_history = []
        state = Board.NONTERMINAL
        while state is Board.NONTERMINAL:
            # TODOJ - choose only 1 random move in a game? need to make the random choice at a uniform stage
            current_move, current_val, is_rand, scores = chooseNextMove(weights, board, epsilon)
            state = board.checkWin(current_move)
            board.makeMove(current_move)
            tmp=(board.boardRep(), current_move, current_val, state)
            board_history.append(tmp)
    
        if DEBUG:
            print()
            print("NEW GAME:")
        
        r,m,v,s = board_history[-1]
        if s is Board.DRAW:
            target_val = 0.5
        else:
            target_val = 1.0
        wins[s]+=1
        
        if save_games:
            # targets are the value of the next nnFunc and only the last value the game value.
            targets=[1.0-v for r,m,v,s in board_history]
            targets=targets[1:] + [target_val]
            board_reps=[r for r,m,v,s in board_history]
#            print(board_history)
#            print(np.hstack(board_reps))
#            print(np.hstack(targets))
#            return None
            
            global games_history
            if 'games_history' not in globals():
                games_history = [[board_reps], [targets]]
            else:
                games_history[0].append(board_reps)
                games_history[1].append(targets)
        
        if learn_weights:
            opt = VanillaOptim(max_processed=1e9, batch_size=0, alpha=alpha)
            for r,m,v,s in reversed(board_history):
                if DEBUG:
                    print("1")
                    print(Board(board_repr=r),m)
                old_val, junk1 = calcVal(weights, r)
                opt.optimize(inp=r, target=target_val, weights=weights)
                new_val, junk1 = calcVal(weights, r)
                if DEBUG:
                    print("val:",v,"target:",target_val,"val_after_update",new_val)
                cost = (new_val - old_val)**2
                if cost > largest_cost_update:
                    largest_cost_update=cost
                target_val = invertVal(new_val,state)
            if (ii+1)%100==0:
                print("num games:",ii+1,"cost:",largest_cost_update, 
                "wins:",wins)
                wins = {Board.OWIN:0,Board.XWIN:0,Board.DRAW:0}
                largest_cost_update = 0    
    print(wins)
            
def actionReward(weights, board, move):
    board.makeMove(move)
    val, rv = calcVal(weights, board.boardRep())
    board.undoMove()
    return val

def calcVal(weights, board_rep):
    rv = valFunc(board_rep, weights)
    val = rv[0][-1]
    return (val, rv)
    
def calcValAndDerivRaw(board_reps, y, weights):
    # forward
    (activations, zs) = nnFunc(board_reps, weights)
    # backprop
    deriv = valFunc_deriv(activations, zs, weights, y)
    return (activations, zs, deriv)

def fastLearn(fast_weights,
              n_cycles = 10,
              n_games = 10000, epsilon = 0.1, 
              n_processed = 1e7, decay = 0.7, alpha = 1, batch_size=0,
              do_plot = True):
    for n in range(n_cycles):
        print("cycle:",n)
        global games_history
        if 'games_history' in globals():
            del games_history
        # global weights
        weights = copy.deepcopy(fast_weights)
        learnXO(weights, reset_weights=False, num_games=n_games, 
                alpha = 0.1, # not in use since learn_weights is False
                epsilon=epsilon, save_games=True,
                learn_weights=False)
                
        dd=createLearningData(games_history)
        
        max_processed = n_processed
        for ii in range(len(dd)):
            print("stage",ii, "data set size",dd[ii][0].shape)
            opt = VanillaOptim(max_processed, batch_size, alpha)
            opt.optimize(inp=dd[ii][0], target=dd[ii][1], weights=fast_weights)
            max_processed *= decay
        
        if do_plot:
            plot_layer(dd, weights)
            
            
def plot_layer(learning_data, weights, nlayers=4):
    fig, axes = plt.subplots(nrows=1, ncols=nlayers, figsize=(45, 10))
    df_line=pd.DataFrame(data={'x':[0,1],'y':[0,1]})
    for layer in range(nlayers):
        aa=learning_data[layer][0]
        bb=learning_data[layer][1]
        jj=np.random.randn(bb.shape[1])/50
        df = pd.DataFrame(data={
            "pred":valFunc(aa,weights)[0][-1].flatten(), 
            "target":bb.flatten() + jj})
        df.plot.scatter(x='pred', y='target',
                        grid=True, title="Layer "+str(layer),
                        xlim=(0,1), ylim=(0,1),
                        ax = axes[layer])
        df_line.plot(legend=False,color="lightgrey", ax = axes[layer])
#    p=ggplot(df2, aes('pred', 'target')) + geom_point()
#    print(p)

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
    
def invertVal(val, state):
    if state is Board.DRAW:
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
    return z*(z>=0)

def sigmaFunc_deriv(z):
    return 1.0*(z>=0)
