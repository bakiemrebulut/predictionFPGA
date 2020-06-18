import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def importDataset(inputSet):
    #####################################################################################################
    ####################################3)import dataset to nparray #####################################
    #####################################################################################################
    print("3)importing dataset")
    ysetc=np.zeros((inputSet*2,1))
    dataset=np.zeros((inputSet*2,784))

    for i in range(inputSet):
        a=str(i)
        ch='shapes/Train/circle/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
            if data[a]<128:
                data[a]=0
            else:
                data[a]=1
        dataset[i]=data
        ysetc[i]=1

    for j in range(inputSet):
        i=j+inputSet
        a=str(j)
        ch='shapes/Train/triangle/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
            if data[a]<128:
                data[a]=0
            else:
                data[a]=1
        dataset[i]=data
    """
    for k in range(inputSet):
        i=k+inputSet*2
        a=str(k)
        ch='shapes/Train/square/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
            if data[a]<128:
                data[a]=0
            else:
                data[a]=1
        dataset[i]=data
    for l in range(inputSet):
        i=l+inputSet*3
        a=str(l)
        ch='shapes/Train/star/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
            if data[a]<128:
                data[a]=0
            else:
                data[a]=1
        dataset[i]=data
    """
    return dataset,ysetc
def importTestset(testNum):
    #####################################################################################################
    ####################################4)import testset to nparray #####################################
    #####################################################################################################
    print("4)importng testset")
    testdataset1=np.zeros((testNum,784))
    testdataset2=np.zeros((testNum,784))
    testdataset3=np.zeros((testNum,784))
    testdataset4=np.zeros((testNum,784))


    testdatasettotal=np.zeros((testNum*3,784))

    for i in range(testNum):
        a=str(i)
        ch='shapes/Test/circle/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
            if data[a]<128:
                data[a]=0
            else:
                data[a]=1
        testdataset1[i]=data

    for j in range(testNum):
        a=str(j)
        ch='shapes/Test/triangle/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
            if data[a]<128:
                data[a]=0
            else:
                data[a]=1
        testdataset2[j]=data
    
    for k in range(testNum):
        a=str(k)
        ch='shapes/Test/star/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
            if data[a]<128:
                data[a]=0
            else:
                data[a]=1
        testdataset3[k]=data
    """for l in range(testNum):
        a=str(l)
        ch='shapes/Test/star/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
            if data[a]<128:
                data[a]=0
            else:
                data[a]=1
        testdataset4[l]=data
    """
    testdatasettotal=np.concatenate((testdataset1,testdataset2,testdataset3))#,testdataset4))
    o=np.ones((testNum,1))
    z=np.zeros((testNum,1))
    testysetc =np.concatenate((o,z,z))#,z,z))    #c
    return testdatasettotal,testysetc

def arrdec2bin(xarr,width):
    xout=np.char.mod('%d',np.zeros(xarr.shape))
    xar=xarr.astype(int)
    xout=np.char.mod('%d',np.remainder(xar,np.ones(xar.shape)*2))
    xar=xar/2
    for i in range(width-1):
        xout=np.char.add(np.char.mod('%d',np.remainder(xar,np.ones(xar.shape)*2)),xout)
        xar=xar/2
    return xout

def dec2bin(x):
    xl=np.trunc(x)
    xr=abs(x)-abs(xl)
    #print(xl,xr)
    #xlb=np.binary_repr(abs(int(xl)), width=6)
    xlb=arrdec2bin(abs(xl),6)
    #sign = '0'  if x >= np.zeros(x.shape) else '1'
    sign=np.char.mod('%d',(np.sign(xl)*(-1)+1)/2)
    #xrb=np.binary_repr(int(xr*256), width=8)
    xrb=arrdec2bin(xr*256,8)
    out=np.char.add(np.char.add(sign,xlb),xrb)
    return out
#####################################################################################################
####################################0)round function          #####################################
#####################################################################################################
def roundbin(x):
    xl=np.trunc(x)
    xr=x-xl
    b=1
    #xrnew=xr
    xrout=np.zeros(xr.shape)
    xrnew3=np.zeros(xr.shape)
    for i in range(8):
        xr=xr*2
        #print(xrnew)
        xrnew3=np.trunc(xr)
        #print("ch:",xrnew3)
        b/=2    
        xrout+=b*xrnew3
        #print("o:",xrout)
        xr-=xrnew3
    return xl+xrout
def roundbinf(x,floats):
    xl=np.trunc(x)
    xr=x-xl
    b=1
    xrnew=xr
    xrout=np.zeros(xr.shape)
    xrnew3=np.zeros(xr.shape)
    for i in range(floats):
        xrnew=xrnew*2
        #print(xrnew)
        xrnew3=np.trunc(xrnew)
        #print("ch:",xrnew3)
        b/=2    
        xrout+=b*xrnew3
        #print("o:",xrout)
        xrnew-=xrnew3
    return xl+xrout
#####################################################################################################
####################################1)sigmoid function          #####################################
#####################################################################################################
def sigmoid(x):
    #threshold
    """
    T1=0
    T2=0
    threshold = np.ones(x.shape)
    threshold[(x < 0)] = 0
    return threshold
    """
    #sigmoid
    #return np.around(1.0/(1 + np.exp(-x)),decimals=2)
    #ramp
    xx=np.zeros(x.shape)
    if len(xx.shape)>1:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i][j]<=-4:
                    xx[i][j]=0
                elif x[i][j]>=4:
                    xx[i][j]=1
                else:
                    xx[i][j]=(x[i][j]+4)/8
    else:
        for i in range(x.shape[0]):
            if x[i]<=-4:
                xx[i]=0
            elif x[i]>=4:
                xx[i]=1
            else:
                xx[i]=(x[i]+4)/8

    return roundbin(xx)
def sigmoidplot():
    x=np.linspace(-10,9.99,num=2000)
    b=sigmoid(x)
    print(b)
    np.savetxt('sigmoidR.out', b, delimiter=' ',fmt='%10.9f')   # 
    #ax = fig.add_subplot(421)
    #plt.plot(x,b)
    #plt.title('sigmoid')
    #plt.show()
#sigmoidplot()
#####################################################################################################
####################################2)neural network classes    #####################################
#####################################################################################################
class testNeuralNetwork:
    def __init__(self, x, w1,w2):
        self.input      = x
        self.weights1   = w1
        self.weights2   = w2                 
        #self.output     = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

class NeuralNetwork:
    def __init__(self, x, y,w1c,w2c):
        self.input      = x
        self.w = 2
        self.weights1   = np.random.rand(self.input.shape[1],self.w)+0.5
        self.weights2   = np.random.rand(self.w,1)+0.5

        self.y          = y
        self.output     = np.zeros(self.y.shape)
        self.w1c        = w1c
        self.w2c        = w2c

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))#(input.shape,w)
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        self.weights1=self.weights1+self.w1c*np.dot(self.input.T,np.dot((self.y-self.output),self.weights2.T))
        self.weights2=self.weights2+self.w2c*np.dot(self.layer1.T,(self.y-self.output))
        self.weights1   = roundbin(self.weights1)
        self.weights2   = roundbin(self.weights2)
def test(nnc,testNum):
    #####################################################################################################
    ####################################6)test NeuralNetwork        #####################################
    #####################################################################################################
    print("6)testing NeuralNetwork")
    tpc=0
    tnc=0
    fpc=0
    fnc=0
    q=3
    testError=np.zeros(testNum*q)

    for i in range(testNum*q):
        testdata = testdatasettotal[i]
        nnTest=testNeuralNetwork(testdata,nnc.weights1,nnc.weights2)
        nnTest.feedforward()
        if i== 10:
            print("circle output:0")
            print(nnTest.output)
        if i== testNum+10:
            print("circle output:1")
            print(nnTest.output)
        if i== 2*testNum+10:
            print("circle output:2")
            print(nnTest.output)
        if i== 3*testNum+10:
            print("circle output:3")
            print(nnTest.output)
        
        testError[i]=abs(nnTest.output-testysetc[i])
        if testysetc[i]==1 and nnTest.output>=0.5:
            tpc+=1
        elif testysetc[i]==1 and nnTest.output<0.5:
            fnc+=1
        elif testysetc[i]==0 and nnTest.output>=0.5:
            fpc+=1
        elif testysetc[i]==0 and nnTest.output<0.5:
            tnc+=1
    true=tpc+tnc
    false=fpc+fnc
    truerate=(true)/(testNum*q)
    print("true/total:",truerate)
    AvgError=np.sum(testError)/(testNum*q)
    print("AvgError::",AvgError)
    fig2 = plt.figure()
    ax = fig2.add_subplot()
    plt.plot(range(testNum*q),testError)
    if truerate>0.98 and AvgError<0.2:
        print("********saving***********")
        print("tpc:",tpc)
        print("tnc:",tnc)
        print("fpc:",fpc)
        print("fnc:",fnc)
        print("")

        print("true:",true)
        print("false:",false)
        print("true/total:",(true)/(testNum*q))
        print("AvgError::",np.sum(testError)/(testNum*q))
        


        print("cw2:")
        print(nnc.weights2)
        print("cw1:")
        print(nnc.weights1)
        
        np.savetxt('circleweights2.out', nnc.weights2, delimiter=' ',fmt='%7.8f')   # 
        np.savetxt('circleweights1.out', nnc.weights1, delimiter=' ',fmt='%7.8f')   # 

        np.save('w2', nnc.weights2)
        np.save('w1', nnc.weights1)
    return AvgError,truerate
def train(nniter,inputSet,nnc):
    #####################################################################################################
    ####################################5.i)train circle NeuralNetwork  #################################
    #####################################################################################################
    print("5.i)training NeuralNetwork   ")
    #X=dataset
    #y = ysetc
    
    grapgRange=4*1
    nnplt=np.zeros((grapgRange,nniter))
    for i in range(nniter):
        nnc.feedforward()
        nnc.backprop()
        #print(nn.output[3].shape)
    
        for j in range(grapgRange):
            nnplt[j][i]=nnc.output[j%2*inputSet]
            
    fig = plt.figure()
    xxx=4
    yyy=1
    for i in range(yyy):
        ax = fig.add_subplot(xxx,yyy,i+yyy*0+1)
        plt.plot(range(nniter),nnplt[i*xxx])
        plt.title("circles")
        ax = fig.add_subplot(xxx,yyy,i+yyy*1+1)
        plt.plot(range(nniter),nnplt[i*xxx+1])
        plt.title("others")
        
        """
        ax = fig.add_subplot(xxx,yyy,i+yyy*2+1)
        plt.plot(range(nniter),nnplt[i*xxx+2])
        plt.title("squares")
        ax = fig.add_subplot(xxx,yyy,i+yyy*3+1)
        plt.plot(range(nniter),nnplt[i*xxx+3])
        plt.title("stars")
        """
    return nnc
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
def bigg(dataset,ysetc,testdatasettotal,testysetc,inputSet,testNum,nnc):
    nnc=train(nniter,inputSet,nnc)
    AvgError,truerate=test(nnc,testNum)
    #w2 = np.load('w2.npy')
    #plt.show()  
    maxx=np.max([np.max(nnc.weights1),np.max(nnc.weights2),abs(np.min(nnc.weights1)),abs(np.min(nnc.weights2))])
    return truerate,AvgError,maxx,nnc

flag=0
count=0
nniter=1000
inputSet=1850
testNum=1850
w1c=0.0001
w2c=0.0008
"""
dataset,ysetc=importDataset(inputSet)     
np.save('dataset', dataset)
np.save('ysetc', ysetc)

testdatasettotal,testysetc=importTestset(testNum)
np.save('testdatasettotal', testdatasettotal)
np.save('testysetc', testysetc)
print("write done..")
"""
dataset=np.load('dataset.npy')
ysetc=np.load('ysetc.npy')
testdatasettotal=np.load('testdatasettotal.npy')
testysetc=np.load('testysetc.npy')
print("read done..")

nnc = NeuralNetwork(dataset,ysetc,w1c,w2c)
a,b,c,nnc=bigg(dataset,ysetc,testdatasettotal,testysetc,inputSet,testNum,nnc)
print(a,b,c)
plt.show()
while a<=0.95 or b>=0.2 or c>=64:
    print("regenerate:")
    w1c=w1c*1.1
    w2c=w2c*1.1
    nnc = NeuralNetwork(dataset,ysetc,w1c,w2c)
    a,b,c,nnc=bigg(dataset,ysetc,testdatasettotal,testysetc,inputSet,testNum,nnc)
    plt.show()  
    print(nnc.w1c,nnc.w2c)
aaa=a
bbb=b
while a!=1.0 or b>=0.01 or c>=64.0:
    print("resume:")
    a,b,c,nnc=bigg(dataset,ysetc,testdatasettotal,testysetc,inputSet,testNum,nnc)
 

    print(nnc.w1c,nnc.w2c)

    print(a,b,c)
    count+=1
    #plt.show()
    if count==100: 
        flag=1
        break
if flag==1:
    while a!=1.0 or b>0.015 or c>=64.0:
        a,b,c,nnc=bigg(dataset,ysetc,testdatasettotal,testysetc,inputSet,testNum,nnc)
        print(a,b,c)
        count+=1
        if count==100: 
            flag=2
            break
if flag==2:
    while a!=1.0 or b>0.1 or c>=64.0:
        a,b,c,nnc=bigg(dataset,ysetc,testdatasettotal,testysetc,inputSet,testNum,nnc)
        print(a,b,c)
        #count+=1
        