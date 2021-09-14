[toc]

# python

## 面向对象
### 封装
内部变量，外部无法访问


### 继承
子类继承父类，子类可使用父类的方法


### 多态
#### 重写
子类可以重写父类的方法，方法名相同
实例化子类，父子都有相同方法名的情况下，调用子类重写后的方法

#### 重载
一个类中定义了多个方法名相同，而参数的数量不同

#### 虚函数
父类只给出方法名，没有给出具体的实现过程，实现过程交给子类


## 多进程
> https://blog.csdn.net/weixin_42134789/article/details/82992326

### Process
```python
from multiprocessing import Process

```



### Pool
```python
from multiprocessing import Pool, cpu_count
```

Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果进程池还没有满，就会创建一个新的进程来执行请求
如果池满，请求就会告知先等待，直到池中有进程结束，才会创建新的进程来执行这些请求


### 进程间通信
进程之间是相互独立的，每个进程都有独立的内存
通过共享内存(nmap模块)，进程之间可以共享对象，使多个进程可以访问同一个内存块
使用队列 queue 来实现不同进程间的通信或数据共享

```python
from multiprocessing import Process, Queue
import os, time, random

# 写数据进程执行的代码:
def write(q):
    print('Process to write: {}'.format(os.getpid()))
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    print('Process to read:{}'.format(os.getpid()))
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    pw = Process(target=write, args=(q, ))
    pr = Process(target=read, args=(q, ))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    pr.terminate()


""""
Process to write: 3036
Put A to queue...
Process to read:9408
Get A from queue.
Put B to queue...
Get B from queue.
Put C to queue...
Get C from queue.
```

## 多线程
threading.Thread 方法可以接收两个参数, 第一个是target，一般指向函数名，第二个时args，需要向函数传递的参数
创建的新线程，调用 start() 方法可让其开始

### 线程同步
> 主线程等待子线程完成后再继续执行，实现同步，我们需要使用join()方法

1. 主线程等待子线程
``` python
import threading
import time

def long_time_task(i):
    print('当前子线程: {} 任务{}'.format(threading.current_thread().name, i))
    time.sleep(2)
    print("结果: {}".format(8 ** 20))

if __name__=='__main__':
    start = time.time()
    print('这是主线程：{}'.format(threading.current_thread().name))
    thread_list = []
    for i in range(1, 3):
        t = threading.Thread(target=long_time_task, args=(i, ))
        thread_list.append(t)

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()

    end = time.time()
    print("总共用时{}秒".format((end - start)))
```

2. 主线程结束时不再执行子线程
``` python
import threading
import time

def long_time_task():
    print('当子线程: {}'.format(threading.current_thread().name))
    time.sleep(2)
    print("结果: {}".format(8 ** 20))

if __name__=='__main__':
    start = time.time()
    print('这是主线程：{}'.format(threading.current_thread().name))
    for i in range(5):
        t = threading.Thread(target=long_time_task, args=())
        t.setDaemon(True)
        t.start()
    end = time.time()
    print("总共用时{}秒".format((end - start)))
```

### 继承 Thread 类重写 run

``` python
import threading
import time

def long_time_task(i):
    time.sleep(2)
    return 8**20

class MyThread(threading.Thread):
    def __init__(self, func, args , name='', ):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
        self.name = name
        self.result = None

    def run(self):
        print('开始子进程{}'.format(self.name))
        self.result = self.func(self.args[0],)
        print("结果: {}".format(self.result))
        print('结束子进程{}'.format(self.name))

if __name__=='__main__':
    start = time.time()
    threads = []
    for i in range(1, 3):
        t = MyThread(long_time_task, (i,), str(i))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    end = time.time()
    print("总共用时{}秒".format((end - start)))

""""
开始子进程1
开始子进程2
结果: 1152921504606846976
结果: 1152921504606846976
结束子进程1
结束子进程2
总共用时2.005445718765259秒
```

### 线程间的数据共享

##### threading.lock() 
线程之间共享数据最大的危险在于多个线程同时改一个变量，其中一个解决方法就是在修改前给其上一把锁lock，threading.lock() 方法可以轻易实现对一个共享变量的锁定

``` python
import threading

class Account:
    def __init__(self):
        self.balance = 0

    def add(self, lock):
        # 获得锁
        lock.acquire()
        for i in range(0, 100000):
            self.balance += 1
        # 释放锁
        lock.release()

    def delete(self, lock):
        # 获得锁
        lock.acquire()
        for i in range(0, 100000):
            self.balance -= 1
            # 释放锁
        lock.release()

if __name__ == "__main__":
    account = Account()
    lock = threading.Lock()
    # 创建线程
   thread_add = threading.Thread(target=account.add, args=(lock,), name='Add')
    thread_delete = threading.Thread(target=account.delete, args=(lock,), name='Delete')

    # 启动线程
   thread_add.start()
    thread_delete.start()

    # 等待线程结束
   thread_add.join()
    thread_delete.join()

    print('The final balance is: {}'.format(account.balance))


```

##### Queue
另一种实现不同线程间数据共享的方法就是使用消息队列queue
创建两个线程，一个负责生成，一个负责消费，生产的产品存放在queue里

```python
from queue import Queue
import random, threading, time

# 生产者类
class Producer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.queue = queue

    def run(self):
        for i in range(1, 5):
            print("{} is producing {} to the queue!".format(self.getName(), i))
            self.queue.put(i)
            time.sleep(random.randrange(10) / 5)
        print("%s finished!" % self.getName())

# 消费者类
class Consumer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.queue = queue

    def run(self):
        for i in range(1, 5):
            val = self.queue.get()
            print("{} is consuming {} in the queue.".format(self.getName(), val))
            time.sleep(random.randrange(10))
        print("%s finished!" % self.getName())

def main():
    queue = Queue()
    producer = Producer('Producer', queue)
    consumer = Consumer('Consumer', queue)

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()
    
    print('All threads finished!')

if __name__ == '__main__':
    main()
```
队列queue的put方法可以将一个对象obj放入队列中，如果队列已满，此方法将等待直到队列有空间可用为止
queue的get方法一次返回队列中的一个成员，如果队列为空，此方法将等待直到队列中有成员可用为止


### Python多进程和多线程
对CPU密集型代码(比如循环计算) - 多进程效率更高
对IO密集型代码(比如文件操作，网络爬虫) - 多线程效率更高

为什么是这样呢？其实也不难理解。对于IO密集型操作，大部分消耗时间其实是等待时间，在等待时间中CPU是不需要工作的，那你在此期间提供双CPU资源也是利用不上的，相反对于CPU密集型代码，2个CPU干活肯定比一个CPU快很多。那么为什么多线程会对IO密集型代码有用呢？这时因为python碰到等待会释放GIL供新的线程使用，实现了线程间的切换。

## GIL




# C++

## 智能指针
动态内存管理经常会出现两种问题：一种是忘记释放内存；一种是尚有指针引用内存的情况下，内存就被释放了，就会产生引用非法内存的指针

- 独占指针，一个内存空间只能由一个指针控制
- 共享指针，多个指针可指向同一内存空间

负责自动释放所指向的对象，指向对象的指针为0时，内存空间被释放



















