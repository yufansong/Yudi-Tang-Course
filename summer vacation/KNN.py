import  numpy as np
import operator

def creatDateSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels =['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDistIndicies[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
group,labels=creatDateSet()
test=classify0([3,0.2],group,labels,3)
print(test)