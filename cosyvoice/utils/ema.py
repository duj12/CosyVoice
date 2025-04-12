import contextlib
from collections import OrderedDict

class EMA:
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        self.model = model
        
        #register model parameters
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.original[name] = param.data
                param.data = self.shadow[name]
            else:
                print(f"Warning: {name} not found in shadow")
    
    def restore(self):
        for name, param in self.model.named_parameters():
            assert name in self.original
            param.data = self.original[name]
        self.original = {}
        
    @contextlib.contextmanager
    def average_parameters(self):
        self.apply_shadow()
        yield
        self.restore()
    
    def state_dict(self):
        return OrderedDict([('shadow', self.shadow)])
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        
    def to(self, device):
        for name, param in self.shadow.items():
            self.shadow[name] = param.to(device)