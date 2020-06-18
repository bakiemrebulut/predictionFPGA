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
    #testdataset3=np.zeros((testNum,784))
    #testdataset4=np.zeros((testNum,784))


    testdatasettotal=np.zeros((testNum*2,784))

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
    """
    for k in range(testNum):
        a=str(k)
        ch='shapes/Test/square/',a,'.png'
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
        testdataset3[i]=data
    for l in range(testNum):
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
        testdataset4[i]=data
    """
    testdatasettotal=np.concatenate((testdataset1,testdataset2))#,testdataset3,testdataset4))
    o=np.ones((testNum,1))
    z=np.zeros((testNum,1))
    testysetc =np.concatenate((o,z))#,z,z))    #c
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
        self.layer1nonsig = np.dot(self.input, self.weights1)
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.outputnonsig = np.dot(self.layer1, self.weights2)
        self.output = sigmoid(np.dot(self.layer1, self.weights2))


def test(testNum,w1,w2,testdatasettotal,testysetc):
    #####################################################################################################
    ####################################6)test NeuralNetwork        #####################################
    #####################################################################################################
    print("6)testing NeuralNetwork")
    tpc=0
    tnc=0
    fpc=0
    fnc=0
    testdata = testdatasettotal[2]
    nnTest=testNeuralNetwork(testdata,w1,w2)
    nnTest.feedforward()
    testError=np.zeros(testNum*2)
    layer1array=np.zeros((testNum*2,nnTest.layer1nonsig.shape[0]))
    outputarray=np.zeros((testNum*2,nnTest.outputnonsig.shape[0]))
    outputreal =np.zeros((testNum*2,nnTest.output.shape[0]))

    for i in range(testNum*2):
        testdata = testdatasettotal[i]
        nnTest=testNeuralNetwork(testdata,w1,w2)
        nnTest.feedforward()
        
        if i== 10:
            print("circle output:0")
            print(nnTest.output)
        if i== testNum+10:
            print("circle output:1")
            print(nnTest.output)
        """if i== 2*testNum+10:
            print("circle output:2")
            print(nnTest.output)
        if i== 3*testNum+10:
            print("circle output:3")
            print(nnTest.output)
        """
        testError[i]=abs(nnTest.output-testysetc[i])
        layer1array[i]=nnTest.layer1nonsig
        outputarray[i]=nnTest.outputnonsig
        outputreal[i]=nnTest.output
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
    truerate=(true)/(testNum*2)
    AvgError=np.sum(testError)/(testNum*2)

    print("tpc:",tpc)
    print("tnc:",tnc)
    print("fpc:",fpc)
    print("fnc:",fnc)
    print("")
    print("true:",true)
    print("false:",false)
    print("true/total:",truerate)
    print("AvgError::",AvgError)
    fig = plt.figure()

    ax = fig.add_subplot(211)
    plt.plot(range(testNum*2),testError)
    ax = fig.add_subplot(212)
    plt.plot(range(testNum*2),outputarray)

    print("----------------------------")
    #print("each lay:",layer1array)
    print("lay_max:",np.max(layer1array))
    print("lay_min:",np.min(layer1array))
    #print("each out:",outputarray)
    print("out_max:",np.max(outputarray))
    print("out_min:",np.min(outputarray))

    #w2 = np.load('w2.npy')

    plt.show()
    return AvgError,truerate


flag=0
count=0
nniter=1000
inputSet=1850
testNum=1850
w1c=0.0001
w2c=0.0008

w1=np.load('w1.npy')
w2=np.load('w2.npy')
testdatasettotal=np.load('testdatasettotal.npy')
testysetc=np.load('testysetc.npy')
print("read done..")
AvgError,truerate=test(testNum,w1,w2,testdatasettotal,testysetc)
#plt.show()
        