import torch
import time


def check_broadcast(x, a, b, c, d):
    return x.max().item() == a.max().item() == b.max().item() == c.max().item() == d.max().item()


def broadcast1(x, a, b, c, d):
    # For example, use a.copy_(x) to copy the data from x to a
    a.copy_(x)
    b.copy_(x)
    c.copy_(x)
    d.copy_(x)


def broadcast2(x, a, b, c, d):
    a.copy_(x)
    b.copy_(a)
    c.copy_(a)
    d.copy_(a)



def broadcast3(x, a, b, c, d):
    sa = torch.cuda.Stream(device='cuda:0')
    sb = torch.cuda.Stream(device='cuda:1')
    sc = torch.cuda.Stream(device='cuda:2')
    sd = torch.cuda.Stream(device='cuda:3')
    
    num_chunks = 32
    chunk_size = (x.numel() + num_chunks - 1) // num_chunks
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, x.numel())
        
        with torch.cuda.stream(sa):
            a.view(-1)[start:end].copy_(x.view(-1)[start:end], non_blocking=True)
        
        with torch.cuda.stream(sb):
            sb.wait_stream(sa)
            b.view(-1)[start:end].copy_(a.view(-1)[start:end], non_blocking=True)
        
        with torch.cuda.stream(sc):
            sc.wait_stream(sa)
            c.view(-1)[start:end].copy_(a.view(-1)[start:end], non_blocking=True)
        
        with torch.cuda.stream(sd):
            sd.wait_stream(sa)
            d.view(-1)[start:end].copy_(a.view(-1)[start:end], non_blocking=True)
    
    sa.synchronize()
    sb.synchronize()
    sc.synchronize()
    sd.synchronize()

def timeit(boradcast, pin_memory=False):

    x = torch.randn((100000, 10000), pin_memory=pin_memory)
    a = torch.zeros((100000, 10000), device="cuda:0")
    b = torch.zeros((100000, 10000), device="cuda:1")
    c = torch.zeros((100000, 10000), device="cuda:2")
    d = torch.zeros((100000, 10000), device="cuda:3")


    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    torch.cuda.synchronize(2)
    torch.cuda.synchronize(3)

    tic = time.time()

    boradcast(x, a, b, c, d)

    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    torch.cuda.synchronize(2)
    torch.cuda.synchronize(3)

    toc = time.time()

    assert check_broadcast(x, a, b, c, d)
    return toc - tic

import tqdm
def avg_timeit(*args):
    num_iterations = 16

    timing_data = []
    for _ in tqdm.tqdm(range(num_iterations)):
        timing_data.append(timeit(*args))

    return sum(timing_data) / num_iterations


print('Running time for broadcast1:', avg_timeit(broadcast1), '(s)')
print('Running time for broadcast2:', avg_timeit(broadcast2), '(s)')
print('Running time for broadcast3:', avg_timeit(broadcast3, True), '(s)')
