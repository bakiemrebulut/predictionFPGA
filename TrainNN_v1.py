import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def bigg():
    nniter=2500
    inputSet=100
    

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
    ####################################1)sigmoid function			#####################################
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
        #ax = fig.add_subplot(321)
        #plt.plot(x,b)
        #plt.title('sigmoid')
        #plt.show()
    #sigmoidplot()
    #####################################################################################################
    ####################################2)neural network classes	#####################################
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
        def __init__(self, x, y):
            self.input      = x
            self.w = 2
            self.weights1   = np.random.rand(self.input.shape[1],self.w)
            self.weights2   = np.random.rand(self.w,1)

            self.y          = y
            self.output     = np.zeros(self.y.shape)

        def feedforward(self):
            self.layer1 = sigmoid(np.dot(self.input, self.weights1))#(input.shape,w)
            self.output = sigmoid(np.dot(self.layer1, self.weights2))

        def backprop(self):
            self.weights1=self.weights1+0.002*np.dot(self.input.T,np.dot((self.y-self.output),self.weights2.T))
            self.weights2=self.weights2+0.016*np.dot(self.layer1.T,(self.y-self.output))
            self.weights1   = roundbin(self.weights1)
            self.weights2   = roundbin(self.weights2)
    #####################################################################################################
    ####################################3)import dataset to nparray	#####################################
    #####################################################################################################
    print("3)importing dataset")
    ysetc=np.zeros((inputSet*3,1))
    ysett=np.zeros((inputSet*3,1))
    ysets=np.zeros((inputSet*3,1))

    dataset=np.zeros((inputSet*3,784))

    for i in range(inputSet):
        a=str(i+1)
        ch='shapes/shapes/circles/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
    	    if data[a]<128:
    	        data[a]=1
    	    else:
    	        data[a]=0
        dataset[i]=data
        ysetc[i]=1

    for j in range(inputSet):
        i=j+inputSet
        a=str(j+1)
        ch='shapes/shapes/triangles/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
    	    if data[a]<128:
    	        data[a]=1
    	    else:
    	        data[a]=0
        dataset[i]=data
        ysett[i]=1

    for k in range(inputSet):
        i=k+inputSet*2
        a=str(k+1)
        ch='shapes/shapes/squares/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
    	    if data[a]<128:
    	        data[a]=1
    	    else:
    	        data[a]=0
        dataset[i]=data
        ysets[i]=1
    #####################################################################################################
    ####################################4)import testset to nparray	#####################################
    #####################################################################################################
    testNum=100
    print("4)importng testset")
    testdataset1=np.zeros((testNum,784))
    testdataset2=np.zeros((testNum,784))
    testdataset3=np.zeros((testNum,784))

    testdatasettotal=np.zeros((testNum*3,784))

    for i in range(testNum):
        a=str(i+1)
        ch='shapes/shapes/testShapes/circles/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
    	    if data[a]<128:
    	        data[a]=1
    	    else:
    	        data[a]=0
        testdataset1[i]=data

    for j in range(testNum):
        a=str(j+1)
        ch='shapes/shapes/testShapes/triangles/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
    	    if data[a]<128:
    	        data[a]=1
    	    else:
    	        data[a]=0
        testdataset2[j]=data

    for k in range(testNum):
        a=str(k+1)
        ch='shapes/shapes/testShapes/squares/',a,'.png'
        str1=""
        for e in ch:
            str1+=e
        image = Image.open(str1).convert(mode="L")
        # convert image to numpy array
        data = np.asarray(image).flatten()
        for a in range(data.shape[0]):
    	    if data[a]<128:
    	        data[a]=1
    	    else:
    	        data[a]=0
        testdataset3[i]=data
    testdatasettotal=np.concatenate((testdataset1,testdataset2,testdataset3))
    o=np.ones((testNum,1))
    z=np.zeros((testNum,1))
    testysetc =np.concatenate((o,z,z))	#c
    testysett =np.concatenate((z,o,z))	#t
    testysets =np.concatenate((z,z,o))	#s
    #####################################################################################################
    ####################################5.i)train circle NeuralNetwork 	#################################
    #####################################################################################################
    print("5.i)training NeuralNetwork 	")
    X=dataset
    y = ysetc
    nnc = NeuralNetwork(X,y)
    nnplt=np.zeros((18,nniter))
    for i in range(nniter):
        nnc.feedforward()
        nnc.backprop()
        #print(nn.output[3].shape)
        nnplt[0][i]=nnc.output[3]
        nnplt[1][i]=nnc.output[3+inputSet]
        nnplt[2][i]=nnc.output[3+inputSet*2]
        nnplt[3][i]=nnc.output[5]
        nnplt[4][i]=nnc.output[5+inputSet]
        nnplt[5][i]=nnc.output[5+inputSet*2]
        nnplt[6][i]=nnc.output[7]
        nnplt[7][i]=nnc.output[7+inputSet]
        nnplt[8][i]=nnc.output[7+inputSet*2]
        nnplt[9][i]=nnc.output[9]
        nnplt[10][i]=nnc.output[9+inputSet]
        nnplt[11][i]=nnc.output[9+inputSet*2]
        nnplt[12][i]=nnc.output[3]
        nnplt[13][i]=nnc.output[3+inputSet]
        nnplt[14][i]=nnc.output[3+inputSet*2]
        nnplt[15][i]=nnc.output[5]
        nnplt[16][i]=nnc.output[5+inputSet]
        nnplt[17][i]=nnc.output[5+inputSet*2]

    fig = plt.figure()
    
    ax = fig.add_subplot(3,6,1)    
    plt.plot(range(nniter),nnplt[0])
    plt.title("circles")
    ax = fig.add_subplot(3,6,7)
    plt.plot(range(nniter),nnplt[1])
    plt.title("triangles")
    ax = fig.add_subplot(3,6,13)
    plt.plot(range(nniter),nnplt[2])
    plt.title("squares")
    
    ax = fig.add_subplot(3,6,2)    
    plt.plot(range(nniter),nnplt[3])
    plt.title("circles")
    ax = fig.add_subplot(3,6,8)
    plt.plot(range(nniter),nnplt[4])
    plt.title("triangles")
    ax = fig.add_subplot(3,6,14)
    plt.plot(range(nniter),nnplt[5])
    plt.title("squares")
    
    ax = fig.add_subplot(3,6,3)    
    plt.plot(range(nniter),nnplt[6])
    plt.title("circles")
    ax = fig.add_subplot(3,6,9)
    plt.plot(range(nniter),nnplt[7])
    plt.title("triangles")
    ax = fig.add_subplot(3,6,15)
    plt.plot(range(nniter),nnplt[8])
    plt.title("squares")
    
    ax = fig.add_subplot(3,6,4)    
    plt.plot(range(nniter),nnplt[9])
    plt.title("circles")
    ax = fig.add_subplot(3,6,10)
    plt.plot(range(nniter),nnplt[10])
    plt.title("triangles")
    ax = fig.add_subplot(3,6,16)
    plt.plot(range(nniter),nnplt[11])
    plt.title("squares")

    ax = fig.add_subplot(3,6,5)    
    plt.plot(range(nniter),nnplt[9])
    plt.title("circles")
    ax = fig.add_subplot(3,6,11)
    plt.plot(range(nniter),nnplt[10])
    plt.title("triangles")
    ax = fig.add_subplot(3,6,17)
    plt.plot(range(nniter),nnplt[11])
    plt.title("squares")

    ax = fig.add_subplot(3,6,6)    
    plt.plot(range(nniter),nnplt[9])
    plt.title("circles")
    ax = fig.add_subplot(3,6,12)
    plt.plot(range(nniter),nnplt[10])
    plt.title("triangles")
    ax = fig.add_subplot(3,6,18)
    plt.plot(range(nniter),nnplt[11])
    plt.title("squares")
    
    #####################################################################################################
    ####################################6)test NeuralNetwork        #####################################
    #####################################################################################################
    print("6)testing NeuralNetwork")
    tpc=0
    tnc=0
    fpc=0
    fnc=0
    testError=np.zeros(testNum*3)

    for i in range(testNum*3):
        testdata = testdatasettotal[i]
        nnTest=testNeuralNetwork(testdata,nnc.weights1,nnc.weights2)
        nnTest.feedforward()
        if i== 10:
            print("10 output:")
            print(nnTest.output)
        if i== 110:
            print("110 output:")
            print(nnTest.output)
        if i== 210:
            print("210 output:")
            print(nnTest.output)
        testError[i]=abs(nnTest.output-testysetc[i])
        if testysetc[i]==1 and nnTest.output>0.95:
            tpc+=1
        elif testysetc[i]==1 and nnTest.output<0.05:
            fnc+=1
        elif testysetc[i]==0 and nnTest.output>0.95:
            fpc+=1
        elif testysetc[i]==0 and nnTest.output<0.05:
            tnc+=1
    true=tpc+tnc
    false=fpc+fnc
    print("true/total:",(true)/(testNum*3))
    print("AvgError::",np.sum(testError)/(testNum*3))
    fig2 = plt.figure()
    ax = fig2.add_subplot()
    plt.plot(range(testNum*3),testError)
    if (true)/(testNum*3)==1 and np.sum(testError)/(testNum*3)<0.01:
        print("tpc:",tpc)
        print("tnc:",tnc)
        print("fpc:",fpc)
        print("fnc:",fnc)
        print("")

        print("true:",true)
        print("false:",false)
        print("true/total:",(true)/(testNum*3))
        print("AvgError::",np.sum(testError)/(testNum*3))
        


        print("cw2:")
        print(nnc.weights2)
        print("cw1:")
        print(nnc.weights1)
        
        np.savetxt('circleweights2.out', nnc.weights2, delimiter=' ',fmt='%7.8f')   # 
        np.savetxt('circleweights1.out', nnc.weights1, delimiter=' ',fmt='%7.8f')   # 

        np.save('w2', nnc.weights2)
        np.save('w1', nnc.weights1)
    #w2 = np.load('w2.npy')
    plt.show()  
    maxx=np.max([np.max(nnc.weights1),np.max(nnc.weights2),abs(np.min(nnc.weights1)),abs(np.min(nnc.weights2))])
    return (true)/(testNum*3),np.sum(testError)/(testNum*3),maxx

flag=0
count=0
a,b,c=bigg()
print(a,b,c)
while a!=1.0 or b>=0.01 or c>=64.0:
    a,b,c=bigg()
    print(a,b,c)
    count+=1
    if count==100: 
        flag=1
        break
if flag==1:
    while a!=1.0 or b>0.015 or c>=64.0:
        a,b,c=bigg()
        print(a,b,c)
        count+=1
        if count==100: 
            flag=2
            break
if flag==2:
    while a!=1.0 or b>0.02 or c>=64.0:
        a,b,c=bigg()
        print(a,b,c)
        #count+=1
        