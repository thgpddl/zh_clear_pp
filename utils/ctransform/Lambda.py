
class Lambda(object):
    def __init__(self,fn):
        self.fn=fn
    def __call__(self,image,force_apply=False):
        return {"image":self.fn(image)}