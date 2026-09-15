"""Approved current-state sensory observations; no policy, planner or action labels."""
import numpy as np
from snake import DIRECTIONS

FEATURES=('food_forward','food_right','obstacle_left','obstacle_forward','obstacle_right')
LOOKAHEAD=4
SPEC=dict(kind='structured_input_anatomical_control',features=list(FEATURES),lookahead_cells=LOOKAHEAD,
    fields=['size','body','direction','food'],food='Egocentric displacement divided by size-1',
    obstacle='Reciprocal distance to first currently occupied body/wall cell within4steps, else0',
    tail='Current tail cell counts as occupied; no prediction of tail departure',
    teacher_or_action_features=False)


def observe(state):
    size=state['size'];body=state['body'];direction=state['direction'];food=state['food']
    if type(size) is not int or size!=16 or type(direction) is not int or direction not in range(4) or not body:
        raise ValueError('Current standard16board, body and original heading convention required')
    body=[tuple(p) for p in body]
    if (len(set(body))!=len(body) or any(len(p)!=2 or any(type(v) not in (int,np.int32,np.int64) for v in p)
            or not all(0<=v<size for v in p) for p in body)):
        raise ValueError('Distinct currently occupied body cells inside the board required')
    x,y=body[0];fx,fy=DIRECTIONS[direction];rx,ry=DIRECTIONS[(direction+1)%4]
    values=np.zeros(5,np.float32)
    if food is not None:
        food=tuple(food)
        if (len(food)!=2 or any(type(v) not in (int,np.int32,np.int64) or not 0<=v<size for v in food) or food in body):
            raise ValueError('Current unoccupied food cell required')
        dx,dy=food[0]-x,food[1]-y
        values[0]=(dx*fx+dy*fy)/(size-1);values[1]=(dx*rx+dy*ry)/(size-1)
    occupied=set(body[1:])
    for index,relative in enumerate((-1,0,1),2):
        dx,dy=DIRECTIONS[(direction+relative)%4]
        for distance in range(1,LOOKAHEAD+1):
            cell=(x+dx*distance,y+dy*distance)
            if not(0<=cell[0]<size and 0<=cell[1]<size) or cell in occupied:
                values[index]=1/distance;break
    return values


def reflected(values):
    """Egocentric reflection for training-only D4 augmentation."""
    values=np.asarray(values)
    if values.shape[-1:]!=(5,):raise ValueError('Five declared sensor observations required')
    result=values.copy();result[...,1]*=-1;result[...,2]=values[...,4];result[...,4]=values[...,2]
    return result
