import sys
import threading
import queue
import torch


def prefetch_map(func, input_iter, prefetch=1, check_interval=5):
    """
    Map a function (func) on a iterable (input_iter), but
    prefetch input values and map them asyncronously as output
    values are consumed.
​
    prefetch: int, the number of values to prefetch asyncronously
    check_interval: int, the number of seconds to block when waiting
                    for output values.
    """
    result_q = queue.Queue(prefetch)
    error_q = queue.Queue(1)
    done_event = threading.Event()

    mapper_thread = threading.Thread(
        target=_mapper_loop, args=(func, input_iter, result_q, error_q, done_event)
    )
    mapper_thread.daemon = True
    mapper_thread.start()

    while not (done_event.is_set() and result_q.empty()):
        try:
            result = result_q.get(timeout=check_interval)
        except queue.Empty:
            continue

        yield result

    if error_q.full():
        raise error_q.get()[1]


def _mapper_loop(func, input_iter, result_q, error_q, done_event):
    try:
        for x in input_iter:
            result = func(x)
            result_q.put(result)
    except BaseException:
        error_q.put(sys.exc_info())
    finally:
        done_event.set()


"""
DEMO usage
"""
# # here is the function to send a batch to the gpu


def prefetch_to_gpu(batch):
    tokens = []
    labels = []
    for x, y in zip(batch[0], batch[1]):
        tokens.append(x.cuda(non_blocking=True))
        labels.append(y.cuda(non_blocking=True))
    return torch.stack(tokens), torch.stack(labels)


# train_loader = srv.MongoBatchDataLoader(...)

# # after creating the train_loader, wrap it in async_map()
# train_loader = prefetch_map(to_gpu, train_loader, prefetch=8)

# # this will cause it to prefetch 8 results from the train_loader and send them to the gpu
