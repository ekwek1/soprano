import torch

RECOGNIZED_DEVICES = ['auto', 'cuda', 'cpu', 'mps']
RECOGNIZED_BACKENDS = ['auto', 'lmdeploy', 'transformers']

def select_device(device='auto'):
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    assert device in RECOGNIZED_DEVICES, f"unrecognized device {device}, device must be in {RECOGNIZED_DEVICES}"
    print(f"Using device {device}")
    return device

def select_backend(backend='auto', device='auto'):
    if backend == 'auto':
        if device == 'cpu':
            backend = 'transformers'
        else:
            try:
                import lmdeploy
                backend = 'lmdeploy'
            except ImportError:
                backend = 'transformers'

    assert backend in RECOGNIZED_BACKENDS, f"unrecognized backend {backend}, backend must be in {RECOGNIZED_BACKENDS}"
    print(f"Using backend {backend}")
    return backend
