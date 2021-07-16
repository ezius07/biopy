import threading
from threading import Event, Thread
from multiprocessing import Queue
from queue import Empty


class AsyncLoader(threading.Thread):
    def __init__(self, q, loader, go, go_next, info):
        super().__init__()
        self.queue = q
        self.info = info
        self.go = go
        self.go_next = go_next
        self.dataloader = loader
                
    def run(self):
        for img, lab in self.dataloader:
            try:
                clearance = self.info.get_nowait()                
                if not clearance:
                    return
            except Empty:
                self.go.wait()
                self.queue.put((img, lab))
                self.go_next.wait()
        self.queue.put((None, None))
                
class ThreadHandler():
    def __init__(self, loaders):
        self.go_next = []
        self.go = []
        self.qs = []
        self.infos = []
        self.threads = []
        self.n = len(loaders)
        for _ in range(self.n):
            self.go_next.append(Event())
            self.go.append(Event())
            self.qs.append(Queue())
            self.infos.append(Queue())
        self.loaders = loaders
        
    def setup_threads(self):
        for i in range(self.n):
            self.threads.insert(i,AsyncLoader(self.qs[i], self.loaders[i], self.go[i], 
                                              self.go_next[i], self.infos[i]))
            
    def start_threads(self):
        for i in range(self.n):
            self.threads[i].start()
    
    def get_from(self, i):
        self.go_next[i].clear()
        self.go[i].set()
        batch, label = self.qs[i].get()
        if batch is None:
            self.threads[i].join()
            self.threads[i] = AsyncLoader(self.qs[i], self.loaders[i], self.go[i], 
                                          self.go_next[i], self.infos[i])

            self.threads[i].start()
            
            self.go_next[i].clear()
            self.go[i].set()
            batch, label = self.qs[i].get()

        self.go[i].clear()
        self.go_next[i].set()

        return batch, label
        
    def terminate_threads(self):
        for i in range(self.n):
            self.infos[i].put(False)
            self.go[i].set()
            self.go_next[i].set()
            self.threads[i].join()
            self.qs[i].close()
            self.qs[i].join_thread()
            self.infos[i].close()
            self.infos[i].join_thread()