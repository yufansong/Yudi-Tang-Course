import multiprocessing as mp
import threading as td
import time

def job(q):
    res=0
    for i in range(10000000):
        res+=i+i**2+i**3
    q.put(res)


def multicore():
    q=mp.Queue()
    p1=mp.Process(target=job,args=(q,))#注意这里的函数是没有括号的，因为是引用不说i调用
    p2=mp.Process(target=job,args=(q,))#注意q这里后面要加逗号，说明是一个可以迭代的东西
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1=q.get()
    res2=q.get()
    print(res1)
    print(res2)
    print('multicore',res1+res2)

def normal():
    res=0
    for _ in range(2):
        for i in range(10000000):
            res += i + i ** 2 + i ** 3
    print('narmal:',res)

def multithread():
    q=mp.Queue()
    t1=td.Thread(target=job,args=(q,))#注意这里的函数是没有括号的，因为是引用不说i调用
    t2=td.Thread(target=job,args=(q,))#注意q这里后面要加逗号，说明是一个可以迭代的东西
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1=q.get()
    res2=q.get()
    print(res1)
    print(res2)
    print('multithread',res1+res2)

if __name__=='__main__':
    st=time.time()
    normal()
    st1=time.time()
    print('normal time:',st1-st)
    multithread()
    st2=time.time()
    print('multithread time:',st2-st)
    multicore()
    print('multicore time:',time.time()-st2)

