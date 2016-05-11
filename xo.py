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

def createBoard():
    board = np.empty([4,4,4],dtype=np.int32)
    board.fill(EMPTY)
    return board

def createBetas():
    # np.random.seed(5)
    beta_shape = [3,4,4,4]
    betas = np.empty(beta_shape,dtype=np.float64)
    betas[...] = 1-2*np.random.random(betas.size).reshape(beta_shape)
    return betas

class Move:
    def __init__(self, _ijk_tuple, _val):
        self.ii=_ijk_tuple[0]
        self.jj=_ijk_tuple[1]
        self.kk=_ijk_tuple[2]
        self.val=_val

def chooseNextMove(board, last_side):
    side = {O:X,X:O}[last_side]
    it = np.nditer(board, flags=['multi_index'])
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
            move, c_val, c_state = chooseNextMove(board, move.val)
        makeMove(board, move)
        val, state = calcVal(board, move)
        side = {O:X,X:O}[side]
    print(board)
    print(state)
    
def getMove(board, side):
    good = False
    while not good:
        mm = input("your move: (enter x,y,z)\n")
        try:
            mm=eval(mm)
            assert( type(mm) is tuple and 
                    len(mm)==3 and 
                    all([type(i) is int and i>=1 and i<=4 for i in mm]))
        except(NameError, AssertionError):
            print("bad input")
            continue
        mm = [i-1 for i in mm]
        if board[mm[0],mm[1],mm[2]]!=EMPTY:
            print("place taken, choose another place")
            continue
        good = True
    move = Move(mm,side)
    return move
            
 
def learnXO(alpha = 0.001, num_games = 10000, reset_betas = False):
    global ALPHA
    ALPHA = alpha
    if 'betas' not in globals() or reset_betas:
        global betas
        betas = createBetas()
    for ii in range(num_games):
        # orig_betas = betas.copy()
        board = createBoard()
        current_move = Move((0,0,0),O) # fake move for initial state
        max_val_diff = 0
        max_betas_diff = 0
        current_move, current_val, state = chooseNextMove(board, current_move.val)
        makeMove(board, current_move)
        while state is NONTERMINAL:
            current_move, current_val, state, val_diff, betas_diff = updateValueFunction(board,current_move,current_val,betas)
            # print(state, '\n', board)
            if abs(val_diff)>max_val_diff:
                max_val_diff=val_diff
            if abs(betas_diff)>max_betas_diff:
                max_betas_diff=betas_diff
        if ii%100==0:
            print('\n', board) 
            print("state=",state,
                  "num_games=",ii,
                  "max_val_diff",max_val_diff,
                  "max_betas_diff=",max_betas_diff)
        
        
def updateValueFunction(board, current_move, current_val, betas):
    opp_move, opp_val, opp_state = chooseNextMove(board, current_move.val)
    makeMove(board, opp_move)
    if opp_state is NONTERMINAL:
        next_move, next_val, next_state = chooseNextMove(board, opp_move.val)
        makeMove(board, next_move)
        new_val = next_val
    else:
        new_val = invertVal(opp_val, opp_state)
        (next_move, next_state) = (opp_move, opp_state)
    prev_betas = betas.copy()
    val_diff = new_val - current_val
    deriv = calcDeriv(board, current_move)
    # print("\nderiv\n",deriv,"\nALPHA\n",ALPHA,"\nval_diff\n",val_diff)
    betas -= ALPHA * val_diff * deriv
    betas_diff = np.abs(prev_betas - betas).sum()
    return (next_move, new_val, next_state, val_diff, betas_diff)
    

def actionReward(board, move):
    makeMove(board, move)
    val, state = calcVal(board, move)
    undoMove(board, move)
    return (val, state)

def makeMove(board, move):
    assert(board[move.ii,move.jj,move.kk]==EMPTY)
    board[move.ii,move.jj,move.kk]=move.val
    
def undoMove(board, move):
    assert(board[move.ii,move.jj,move.kk]!=EMPTY)
    board[move.ii,move.jj,move.kk]=EMPTY
    
def calcVal(board, last_move):
    win = checkWin(board, last_move)
    if win:
        final_state={O:OWIN,X:XWIN}[last_move.val]
        return (1.0,final_state)
    if (board==EMPTY).sum()==0:
        return (0.0, DRAW)
    board_rep = boardRep(board, last_move.val)
    val = valFunc(board_rep, betas)
    return (val, NONTERMINAL)
    
def calcDeriv(board, last_move):
    board_rep = boardRep(board, last_move.val)
    deriv = valFuncDeriv(board_rep, betas)
    return deriv
    
def invertVal(val, state):
    if state is DRAW:
        return val
    return 1.0 - val             
        
def boardRep(board, side):
    other_side = {O:X,X:O}[side]
    return np.stack((board==side, board==other_side, board==EMPTY))
    
def valFunc(x, betas):
    val = 1.0/(1.0+np.exp((x*betas).sum()))
    return val

def valFuncDeriv(x, betas):
    xb_sum = np.exp((x*betas).sum())
    val = (-np.power(1.0+xb_sum,-2)*xb_sum)*x 
    return val

    
def checkWin(board, last_move):
    val = last_move.val
    ii = last_move.ii
    jj = last_move.jj
    kk = last_move.kk
    won = True
    for nn in range(0,BSIZE):
        if board[ii,jj,nn]!=val:
            won = False
            break
    if won:
        return True
    won = True
    for nn in range(0,BSIZE):
        if board[ii,nn,kk]!=val:
            won = False
            break
    if won:
        return True
    won = True
    for nn in range(0,BSIZE):
        if board[nn,jj,kk]!=val:
            won = False
            break
    if won:
        return True
    if ii==jj:
        won = True
        for nn in range(0,BSIZE):
            if board[nn,nn,kk]!=val:
                won = False
                break
        if won:
            return True
    if ii==BSIZE-1-jj:
        won = True 
        for nn in range(0,BSIZE):
            if board[nn,BSIZE-1-nn,kk]!=val:
                won = False
                break
        if won:
            return True
    if ii==kk:
        won = True
        for nn in range(0,BSIZE):
            if board[nn,jj,nn]!=val:
                won = False
                break
        if won:
            return True
    if ii==BSIZE-1-kk:
        won = True 
        for nn in range(0,BSIZE):
            if board[nn,jj,BSIZE-1-nn]!=val:
                won = False
                break
        if won:
            return True
    if jj==kk:
        won = True
        for nn in range(0,BSIZE):
            if board[ii,nn,nn]!=val:
                won = False
                break
        if won:
            return True
    if jj==BSIZE-1-kk:
        won = True 
        for nn in range(0,BSIZE):
            if board[ii,nn,BSIZE-1-nn]!=val:
                won = False
                break
        if won:
            return True
    if ii==jj and jj==kk:
        won = True
        for nn in range(0,BSIZE):
            if board[nn,nn,nn]!=val:
                won = False
                break
        if won:
            return True
    if ii==jj and jj==BSIZE-1-kk:
        won = True
        for nn in range(0,BSIZE):
            if board[nn,nn,BSIZE-1-nn]!=val:
                won = False
                break
        if won:
            return True
    if ii==BSIZE-1-jj and jj==kk:
        won = True
        for nn in range(0,BSIZE):
            if board[nn,BSIZE-1-nn,nn]!=val:
                won = False
                break
        if won:
            return True
    if ii==BSIZE-1-jj and jj==BSIZE-1-kk:
        won = True
        for nn in range(0,BSIZE):
            if board[nn,BSIZE-1-nn,BSIZE-1-nn]!=val:
                won = False
                break
        if won:
            return True
    return False
    
