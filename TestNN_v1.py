import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
nniter=5000
inputSet=100
fig = plt.figure(figsize=(12,8))
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
                if x[i][j]<-4:
                    xx[i][j]=0
                elif x[i][j]>4:
                    xx[i][j]=1
                else:
                    xx[i][j]=(x[i][j]+4)/8
    else:
        for i in range(x.shape[0]):
            if x[i]<-4:
                xx[i]=0
            elif x[i]>4:
                xx[i]=1
            else:
                xx[i]=(x[i]+4)/8

    return roundbin(xx)
def sigmoidplot():
    x=np.linspace(-10,9.99,num=2000)
    b=sigmoid(x)
    print(b)
    np.savetxt('sigmoidR.out', b, delimiter=' ',fmt='%10.9f')   # 
    ax = fig.add_subplot(321)
    plt.plot(x,b)
    plt.title('sigmoid')
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

    def feedforward(self):
        self.layer1nonsig = np.dot(self.input, self.weights1)
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))

        self.outputnonsig = np.dot(self.layer1, self.weights2)
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

#####################################################################################################
####################################4)import testset to nparray	#####################################
#####################################################################################################
"""testNum=35
print("4)importng testset")

testdatasettotal=np.zeros((testNum*10,784))

for i in range(10):
	for j in range(testNum):
	    a=str(j+1)
	    b=str(i)
	    #ch='shapes/shapes/testShapes/circles/',a,'.png'
	    ch='mnist/trainingSet/trainingSet/',b,'/',a,'.jpg'    
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
	    testdatasettotal[j+i*10]=data

o=np.ones((testNum,1))
z=np.zeros((testNum*9,1))
testysetc =np.concatenate((o,z))	#c

"""
testNum=100
print("4)importng testset")
testdataset1=np.zeros((testNum,784))
testdataset2=np.zeros((testNum,784))
testdataset3=np.zeros((testNum,784))

testdatasettotal=np.zeros((testNum*3,784))

for i in range(testNum):
    a=str(i+1)
    ch='mnist/Train/0/',a,'.jpg'#'shapes/shapes/testShapes/circles/',a,'.png'
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
####################################6)test NeuralNetwork        #####################################
#####################################################################################################
print("6)testing NeuralNetwork")
tpc=0
tnc=0
fpc=0
fnc=0
testError=np.zeros(testNum*3)


weights1 = np.load('w1.npy')
weights2 = np.load('w2.npy')
testdata = testdatasettotal[12]
nnTest=testNeuralNetwork(testdata,weights1,weights2)
nnTest.feedforward()
layer1array=np.zeros((testNum*3,nnTest.layer1nonsig.shape[0]))
outputarray=np.zeros((testNum*3,nnTest.outputnonsig.shape[0]))
outputreal =np.zeros((testNum*3,nnTest.output.shape[0]))

print(weights1.shape)
print(weights2.shape)


for i in range(testNum*3):
    testdata = testdatasettotal[i]
    nnTest=testNeuralNetwork(testdata,weights1,weights2)
    nnTest.feedforward()
    if i== 10:
        print("circle output:10")
        print(nnTest.output)
    if i== 110:
        print("circle output:110")
        print(nnTest.output)
    if i== 210:
        print("circle output:210")
        print(nnTest.output)
    testError[i]=abs(nnTest.output-testysetc[i])
    layer1array[i]=nnTest.layer1nonsig
    outputarray[i]=nnTest.outputnonsig
    outputreal[i]=nnTest.output
    if testysetc[i]==1 and nnTest.output>0.9:
        tpc+=1
    elif testysetc[i]==1 and nnTest.output<0.1:
        fnc+=1
    elif testysetc[i]==0 and nnTest.output>0.9:
        fpc+=1
    elif testysetc[i]==0 and nnTest.output<0.1:
        tnc+=1

print("tpc:",tpc)
print("tnc:",tnc)
print("fpc:",fpc)
print("fnc:",fnc)
print("")
true=tpc+tnc
false=fpc+fnc
print("true:",true)
print("false:",false)
print("true/total:",(true)/(testNum*3))
print("AvgError::",np.sum(testError)/(testNum*3))
ax = fig.add_subplot(211)
plt.plot(range(testNum*3),testError)
ax = fig.add_subplot(212)
plt.plot(range(testNum*3),outputreal)

print("----------------------------")
#print("each lay:",layer1array)
print("lay_max:",np.max(layer1array))
print("lay_min:",np.min(layer1array))
#print("each out:",outputarray)
print("out_max:",np.max(outputarray))
print("out_min:",np.min(outputarray))

#w2 = np.load('w2.npy')

plt.show()

"""
print("6)testing NeuralNetwork")
tpc=0
tnc=0
fpc=0
fnc=0
testError=np.zeros(testNum*10)
outputGraph=np.zeros(testNum*10)

weights1 = np.load('w1.npy')
weights2 = np.load('w2.npy')
print(weights2)
print(weights1)
for i in range(testNum*10):
	testdata = testdatasettotal[i]
	nnTest=testNeuralNetwork(testdata,weights1,weights2)
	nnTest.feedforward()
	if i== 10:
		print("circle output:10  0")
		print(nnTest.output)
	if i== testNum+10:
		print("circle output:410  1")
		print(nnTest.output)
	if i== testNum*2+10:
		print("circle output:810   2")
		print(nnTest.output)
	if i== testNum*3+10:
		print("circle output:810   3")
		print(nnTest.output)
	if i== testNum*4+10:
		print("circle output:810   4")
		print(nnTest.output)
	if i== testNum*5+10:
		print("circle output:810   5")
		print(nnTest.output)
	if i== testNum*6+10:
		print("circle output:810   6")
		print(nnTest.output)
	if i== testNum*7+10:
		print("circle output:810   7")
		print(nnTest.output)
	if i== testNum*8+10:
		print("circle output:810   8")
		print(nnTest.output)
	if i== testNum*9+10:
		print("circle output:810   9")
		print(nnTest.output)
	outputGraph[i]=nnTest.output
	testError[i]=abs(nnTest.output-testysetc[i])
	if testysetc[i]==1 and nnTest.output>0.9:
		tpc+=1
	elif testysetc[i]==1 and nnTest.output<0.1:
		fnc+=1
	elif testysetc[i]==0 and nnTest.output>0.9:
		fpc+=1
	elif testysetc[i]==0 and nnTest.output<0.1:
		tnc+=1
print("tpc:",tpc)
print("tnc:",tnc)
print("fpc:",fpc)
print("fnc:",fnc)
print("")
true=tpc+tnc
false=fpc+fnc
print("true:",true)
print("false:",false)
print("true/total:",(true)/(testNum*10))
print("AvgError::",np.sum(testError)/(testNum*10))
ax = fig.add_subplot(325)
plt.plot(range(testNum*10),testError)
plt.title("testError")
ax = fig.add_subplot(326)
plt.plot(range(testNum*10),outputGraph)
plt.title("outputs")

plt.show()


"""