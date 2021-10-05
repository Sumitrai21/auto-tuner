


class Train():
    def __init__(self,cfg,trainloader,model):
        self.cfg = cfg
        self.trainloader = trainloader
        self.model = model



    def begin_training(self):
        for k,v in model.named_parameters():
            v.requires_grad = True
            if any(x in k for x in freeze):
                v.requires_grad=False
