import torch

# Monkey patch the __getattr__ method in torch._classes
original_getattr = torch._classes.__getattr__

def safe_getattr(self, attr):
    if attr == "__path__":
        return None
    return original_getattr(self, attr)

torch._classes.__getattr__ = safe_getattr
