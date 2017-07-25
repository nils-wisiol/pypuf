from numpy import zeros,array,sign,dot,random
from sys import stderr

def getChallenge(wArray, minAccuracy):
    posArray=[]
    negArray=[]
    challengeArray=zeros(wArray.size)
    sumPos=0
    sumNeg=0
    weightArray=[]
    for i in range(wArray.size):
        weightArray.append((wArray[i],i))
    weightArray=sorted(weightArray, key=lambda tup: abs(tup[0]), reverse=True)
#    print(weightArray)
    "greedy algorithm"
    for i in range(len(weightArray)):
        if sumPos<=sumNeg:
            posArray.append((abs(weightArray[i][0]),weightArray[i][1]))
            sumPos+=abs(weightArray[i][0])
            challengeArray[weightArray[i][1]]=1*__signum(weightArray[i][0])
        else:
            negArray.append((abs(weightArray[i][0]),weightArray[i][1]))
            sumNeg+=abs(weightArray[i][0])
            challengeArray[weightArray[i][1]]=-1*__signum(weightArray[i][0])
#    print(challengeArray)        
    actAccuracy=abs(sumPos-sumNeg)
#    print(sumPos)
#    print(sumNeg)
#    print(actAccuracy)
#    print(minAccuracy)
#    print("")
#    if actAccuracy<0.2*minAccuracy:
#        print("juchu")
#        return challengeArray
    
    
#    print(posArray)
#    print(negArray)
    
    optSwap=(sumPos-sumNeg)/2.0
#    print("optSwap: %s"% optSwap)
    "heuristik"

    while True:#actAccuracy>0.2*minAccuracy:
        posIndex=-1
        negIndex=-1
        bestPosIndex=0
        bestNegIndex=0
        bestFit=[0,0,float('Inf')]
        swapVal=0
        while ((posIndex<len(posArray))&(negIndex<len(negArray))):
            if posIndex>=0:
                memoPos=posArray[posIndex][0]
            else:
                memoPos=0
            if negIndex>=0:
                memoNeg=negArray[negIndex][0]
            else:
                memoNeg=0
            
            swapVal=memoPos-memoNeg
#            print(memoPos)
#            print(swapVal)
#            print("posibble swap: %s" % swapVal)
#            print(posArray[posIndex])
#            print(negArray[negIndex])
#            print("")
            if abs(optSwap-swapVal)<abs(bestFit[2]):
                bestFit[0]=posArray[posIndex]
                bestFit[1]=negArray[negIndex]
                bestFit[2]=(optSwap-swapVal)
                bestPosIndex=posIndex
                bestNegIndex=negIndex
            if optSwap-swapVal>0:
                posIndex+=1
            else:
                negIndex+=1
                
#        print("Accuracy after chosen swap: %s" % (abs(bestFit[2])*2))
#        print("posIndex: %s" % bestPosIndex)
#        print("negIndex: %s" % bestNegIndex)
        if(abs(bestFit[2])*2<actAccuracy):
            [posArray,negArray,challengeArray]=__swap(posArray,negArray,bestPosIndex,bestNegIndex,bestFit,challengeArray)
            optSwap=bestFit[2]
            actAccuracy=abs(bestFit[2])*2;
#            print("opt swap: %s" %optSwap)
#            print(posArray)
#            print(negArray)
#            print(challengeArray)
        else:
            break;
        
        
    if challengeArray[len(weightArray)-1]<0:
        challengeArray=-1*challengeArray
    return challengeArray

def __swap(posArray,negArray,posIndex,negIndex,bestFit, challengeArray):
    if(posIndex>=0):
        newNeg = posArray.pop(posIndex)
    if(negIndex>=0):
        newPos = negArray.pop(negIndex)
        
    if(posIndex>=0):
        if(len(negArray)==0): #special case, where negArray tries to swap one and only element
            negArray.insert(0,newNeg)
            challengeArray[bestFit[1][1]]*=-1
        else:
            for i in range(len(negArray)):
                if ((i==len(negArray)-1)|(newNeg[0]<negArray[i][0])):
                    negArray.insert(i,newNeg)
                    challengeArray[bestFit[0][1]]*=-1
                    break;
    if(negIndex>=0):        
        if(len(posArray)==0): #special case, where posArray tries to swap one and only element
            posArray.insert(0,newPos)
            challengeArray[bestFit[0][1]]*=-1
        else:
            for i in range(len(posArray)):
                if ((i==len(posArray)-1)|(newPos[0]<posArray[i][0])):
#                    print(i)
#                    print(i==len(posArray)-1)
#                    print(newPos[0]>posArray[i][0])
#                    print(newPos[0])
#                    print(posArray[i][0])
                    posArray.insert(i,newPos)
                    challengeArray[bestFit[1][1]]*=-1
                    break;
#    print(posArray)
#    print(negArray)
    return [posArray,negArray, challengeArray]

def __signum(x):
    if sign(x)!=0:
        return sign(x)
    else:
        return -1

def __getKey(x):
    return x[0]

#weightArray=array([1,2,2,3,0.45,0.3,4.5,-3,3,-12,20])
#minAccuracy=0.01
#actAccuracy=0
#challengeArray=getChallenge(weightArray,minAccuracy)
#print(challengeArray)
#print("%s\n" % dot(weightArray,challengeArray))