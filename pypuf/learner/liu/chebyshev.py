import pulp

from numpy import array,ones,zeros,sqrt

"""
this implementation can be compared to the computation of the chebyshevcenter but 
can be more accuratly be described as the largest ball, fitting into the inside of a 
(not bounded) polytope, given by some halfspaces a1*x1+ ... + an*xn <= 0 or > 0   
which center lies in the {-1,1}^n-cube
:param challenges: the halfspaces, given as a non-empty list of n-dimensional numpy-arrays
:param reponses: a list, containing the information about the inequality (-1: <= ; 1:>)
:return [center,radius]: the center of the given polytope
"""
def findCenter(challenges, responses):
    """standardize the halfspaces to ...<=0"""
    for i in range(len(responses)):
        if(responses[i]>0):
            challenges[i]=-1*challenges[i]
            responses[i]=-1*responses[i]
    
    n=challenges[0].shape[0]   
    listOfVariables=[]
    for i in range(1,n+1):
        name=(str(i))
        listOfVariables.append(pulp.LpVariable(name,-1,1,pulp.LpContinuous))
    
    prob = pulp.LpProblem('center',pulp.LpMaximize)
    radius=pulp.LpVariable(str(0),None,None,pulp.LpContinuous)
    
    """minimize radius"""
    prob+=radius 
    for i in range(len(challenges)):
        for j in range(n):
            if j==0:
                memo=sqrt(n)*radius
            memo+=challenges[i][j]*listOfVariables[j]
        """adding inequaltiy"""
        prob+=memo<=0    
    prob.solve()
    radius=prob.variables()[0].value()
    center=zeros(n)
    for i in range(1,n):
        center[int(prob.variables()[i].name)-1]=prob.variables()[i].value()
    return [center,radius]

#def __signum(x):
#    if sign(x)!=0:
#        return sign(x)
#    else:
#        return -1

#weight_array=array([1,3,-5,7,20])
#challenges=[]
#responses=[]
#challenges.append(array([1,1,1,1,1]))
#responses.append(__signum(sum(weight_array*challenges[0])))

#print(challenges)
#print(responses)
#findCenter(challenges,responses)