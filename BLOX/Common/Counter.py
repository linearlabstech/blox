

class Cntr:

    def __init__(self,init_value=0.):
        self.cnt = init_value

    def __iadd__(self,i):
        if isinstance(i,type(self.cnt) ):self.cnt+=i
        if isinstance(i,Cntr):self.cnt+=i
        return self

    def __add__(self,i):
        if isinstance(i,type(self.cnt) ):self.cnt+=i
        if isinstance(i,Cntr):self+=i
        return self

    def __truediv__(self,v):
        if isinstance(i,type(self.cnt) ):self.cnt/=v
        if isinstance(i,Cntr):self/=v
        return self
    
    def __int__(self):
        return int(self.cnt)
    def __float__(self):
        return float(self.cnt)
