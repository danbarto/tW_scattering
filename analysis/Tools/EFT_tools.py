import copy

def make_scan(operator='ctW', C_min=-10, C_max=10, step=1, is2D=False):
    points = []
    C_min_tmp = int(C_min/step) if step<1 else C_min
    C_max_tmp = int(C_max/step) if step<1 else C_max
    step_tmp = 1 if step<1 else step
    
    if is2D:
        temp = [0,0]
        operators = ['cpt', 'cpQM']
    else:
        temp = [0,0,0,0,0,0]
        operators = ['ctZ', 'cpt', 'cpQM', 'cpQ3', 'ctW', 'ctp']
    
    for x in range(C_min_tmp, C_max_tmp+1, step_tmp):
        point = copy.deepcopy(temp)
        point[operators.index(operator)] = x*step if step < 1 else x
        points.append({'name': '%s_%.1f'%(operator, x*step if step < 1 else x), 'point': point})
        
    return points

def make_scan_2D(operators=['cpt','cpQM'], C_min=-10, C_max=10, step=1):
    points = []
    C_min_tmp = int(C_min/step) if step<1 else C_min
    C_max_tmp = int(C_max/step) if step<1 else C_max
    step_tmp = 1 if step<1 else step

    temp = [0,0]
    for x1 in range(C_min_tmp, C_max_tmp+1, step_tmp):
        point_tmp = copy.deepcopy(temp)
        point_tmp[0] = x1*step if step < 1 else x1
        points_tmp = []
        for x2 in range(C_min_tmp, C_max_tmp+1, step_tmp):
            point = copy.deepcopy(point_tmp)
            point[1] = x2*step if step < 1 else x2
            points_tmp.append({'name': '%s_%.1f_%s_%.1f'%(operators[0], x1*step if step < 1 else x1, operators[1], x2*step if step < 1 else x2), 'point': point})
        points.append(points_tmp)

    return points
