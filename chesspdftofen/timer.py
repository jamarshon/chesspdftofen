import time

class Timer(object):
  start = None

  @classmethod
  def begin(self):
    self.start = time.time()

  @classmethod
  def end(self):
    return str(time.time() - start)