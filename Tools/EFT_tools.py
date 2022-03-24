
import copy
def make_scan(operator='ctW', C_min=-10, C_max=10, step=1):
    points = []
    C_min_tmp = int(C_min/step) if step<1 else C_min
    C_max_tmp = int(C_max/step) if step<1 else C_max
    step_tmp = 1 if step<1 else step
    
    temp = [0,0,0,0,0,0]
    operators = ['ctZ', 'cpt', 'cpQM', 'cpQ3', 'ctW', 'ctp']
    
    for x in range(C_min_tmp, C_max_tmp+1, step_tmp):
        point = copy.deepcopy(temp)
        point[operators.index(operator)] = x*step if step < 1 else x
        points.append({'name': '%s_%.1f'%(operator, x*step if step < 1 else x), 'point': point})
        
    return points
