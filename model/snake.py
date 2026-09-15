"""Deterministic, wall-bounded monochrome Snake inspired by late-1990s phones."""
import random
import numpy as np
DIRECTIONS=((0,-1),(1,0),(0,1),(-1,0))
class Snake:
 def __init__(self,seed=97,size=16):
  self.rng=random.Random(seed);self.size=size;self.reset()
 def reset(self):
  m=self.size//2;self.body=[(m,m),(m-1,m),(m-2,m)];self.direction=1;self.score=0;self.moves=0;self.since_food=0;self.done=False;self.reason='';self.spawn_food()
 def spawn_food(self):
  empty=[(x,y) for y in range(self.size) for x in range(self.size) if (x,y) not in self.body]
  self.food=self.rng.choice(empty) if empty else None
  if not empty:self.done=True;self.reason='Board complete'
 def distance(self):return sum(abs(a-b) for a,b in zip(self.body[0],self.food)) if self.food else 0
 def step(self,action):
  if self.done:return 0.,True
  old_distance=self.distance();self.direction=(self.direction+action-1)%4;dx,dy=DIRECTIONS[self.direction];x,y=self.body[0];head=(x+dx,y+dy);eating=head==self.food
  # Entering the departing tail cell is legal on a non-growth move.
  occupied=self.body if eating else self.body[:-1];self.moves+=1;self.since_food+=1
  if not(0<=head[0]<self.size and 0<=head[1]<self.size) or head in occupied:
   self.done=True;self.reason='Collision';return -10.,True
  self.body.insert(0,head)
  if eating:self.score+=1;self.since_food=0;self.spawn_food();reward=10.
  else:
   self.body.pop();reward=-.01+.1*(old_distance-self.distance())
  if self.since_food>4*self.size*self.size:self.done=True;self.reason='No food for too long';reward=-5.
  return reward,self.done
 def pixels(self):
  p=np.zeros((self.size,self.size),np.float32)
  for x,y in self.body:p[y,x]=.5
  x,y=self.body[0];p[y,x]=.8
  if self.food:x,y=self.food;p[y,x]=1
  return p
 def state(self):return dict(size=self.size,body=self.body,food=self.food,direction=self.direction,score=self.score,moves=self.moves,done=self.done,reason=self.reason)
