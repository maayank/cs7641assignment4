import numpy as np

def make_lake(S=64, p_slip=.1, p_hole=.1, seed=None):
    '''
    0 - Up
    1 - Left
    2 - Down
    3 - Right
    '''
    oldS = S
    S = int(np.sqrt(S))
    assert S * S == oldS
    if seed is not None:
        np.random.seed(seed)
    def get_surrounding_coords(x, y):
        res = []
        for x_d in [-1, 0, 1]:
            new_x = x + x_d
            if new_x < 0 or new_x >= S:
                continue

            for y_d in [-1, 0, 1]:
                if x_d == 0 and y_d == 0:
                    continue

                new_y = y + y_d
                if new_y < 0 or new_y >= S:
                    continue
                
                res.append((new_x, new_y))
        return res
    
    def get_target_coord(x, y, a):
        if a == 0:
            x_d = -1
        elif a == 2:
            x_d = 1
        else:
            x_d = 0

        if a == 1:
            y_d = -1
        elif a == 3:
            y_d = 1
        else:
            y_d = 0

        
        new_x = x + x_d
        if new_x < 0 or new_x >= S:
            return None

        new_y = y + y_d
        if new_y < 0 or new_y >= S:
            return None
        
        return (new_x, new_y)
    
    def get_p_assignments(x, y, a):
        if x == S - 1 and y == S - 1:
            return {(0,0): 1.0}
        p_move = 1.0 - p_slip
        p_slip_single = p_slip / 7 # Everywhere but current place and target
        target = get_target_coord(x, y, a)
        unused_slips = 7
        used_move = False
        
        coord2p = {}
        for c in get_surrounding_coords(x,y):
            if c == target:
                p = p_move
                used_move = True
            else:
                p = p_slip_single
                unused_slips -= 1
            coord2p[c] = p
        origin_p = 0
        if unused_slips:
            origin_p += unused_slips * p_slip_single
        if not used_move:
            origin_p += p_move
        coord2p[(x,y)] = origin_p

#        s = sum(coord2p.values())
#        if s < 1:
#            print(s)
#            print(f'{a}: {x}, {y}')
#            quit()
        return coord2p

    all_mat = []
    for a in range(4):
        action_mat = []
        for x in range(S):
            for y in range(S):
                mat = np.zeros((S,S))
                coord2p = get_p_assignments(x,y,a)
                for c, p in coord2p.items():
                    mat[c] = p
                action_mat.append(mat.reshape(S*S))
        all_mat.append(action_mat)
    P = np.asarray(all_mat)
    R = np.zeros(S*S)
    for i in range(1, S*S):
        roll = np.random.uniform(0, 1)
        if roll < p_hole:
            R[i] = -100
            newRSlice = np.zeros_like(P[:, i, :])
            newRSlice[:,0] = 1
            P[:, i, :] = newRSlice
    R[-1] = 1000
    return P, R

if __name__ == '__main__':
    print(make_lake(seed=42))