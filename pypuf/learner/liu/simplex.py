# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 21:13:01 2017

@author: alpha51
"""
from math import pow
from numpy import zeros,sqrt,array,abs,round
from sys import stderr

class AdjustedSimplexAlg():
    
    def __init__(self):
        self.tolerance=pow(0.1,20)
        self.roundDigits=10
    
    def solve(self, challenges,responses):
        """standardize the halfspaces to ...<=0"""
        for i in range(len(responses)):
               if(responses[i]>0):
                   challenges[i]=-1*challenges[i]
                   responses[i]=-1*responses[i]
        
        """dimension"""
        n=challenges[0].shape[0]
        """number of challenges"""
        m=len(challenges)
        
        """
        erstelle simplextableau in der form
        
        z | r a11 a12 a21 a22 ... an1 an2 | s1 s2 s3 ... s(m+2n) | b
        
        wobei ai = ai1 - ai2
        """
        columns= 1 + (2*n+1) + (m+2*n) + 1
        rows=m+2*n+1
        
        simplextableau=[]
        
        for i in range(0,m):
            row = zeros(columns)
            row[0]=0
            row[1]=sqrt(n)
            for j in range(0,n):
                row[2+(2*j)]   = challenges[i][j]
                row[2+(2*j)+1] = challenges[i][j] * (-1)
            row[2+(2*n)+i]=1 
            simplextableau.append(row)
                
        for i in range(0,n):
            row = zeros(columns)
            row[2+(2*n)+ m + i] = 1
            row[2+(2*i)] = 1
            row[2+(2*i)+1] = -1
            row[2+(2*n)+m+2*n] = 1
            simplextableau.append(row)
        for i in range(0,n):
            row = zeros(columns)
            row[2+(2*n)+ m + n + i] = 1
            row[2+(2*i)] = -1
            row[2+(2*i)+1] = 1
            row[2+(2*n)+m+2*n] = 1
            simplextableau.append(row)
            
        row=zeros(columns)
        row[0]=1
        row[1]=-1
        simplextableau.append(row)
        
        
        while True:
            [pivotColumn,minimum]=self.__searchPivotCol(simplextableau)
            
#            for i in range(rows):
#                stderr.write("\n")
#                for j in range(columns):
#                    stderr.write("%f " % simplextableau[i][j])
#            stderr.write("\n")
#            stderr.write("pivotColumn: %d \n" % pivotColumn)
            
            if minimum >= -1*self.tolerance:
                #self.printTableau(simplextableau)
                return self.__getSolution(simplextableau,n)
            
            unbounded=self.__checkPivotCol(simplextableau,pivotColumn)
            if unbounded:
                print("unbounded\n")
                return None
            
            [pivotRow,quotient]=self.__searchPivotRow(simplextableau,pivotColumn)
            
            simplextableau=self.__simplexStep(simplextableau,pivotColumn,pivotRow)
        
#            """----------------------------------------------------"""
#            stderr.write("minimum: %f \n" % minimum)
#            stderr.write("pivotrow: %d \n" % pivotRow)
#            stderr.write("quotient: %f \n" % quotient)
            
#            stderr.write("ZFZ:")
#            for i in range(columns):
#                stderr.write("%f | " % simplextableau[-1][i])
#            stderr.write("\n")
#            """---------------------------------------------------"""
        
    def __searchPivotCol(self,tableau):
        row=tableau[-1]
        minimum=float("inf")
        column=-1
        for i in range(1,row.size-1):
            if(row[i]<minimum):
                minimum=row[i]
                column=i
        return [column,minimum]
    
    def __checkPivotCol(self,tableau,i):
        unbounded=True
        for j in range(len(tableau)):
            if tableau[j][i]>0:
                unbounded=False
        if unbounded == True:
            for l in range(len(tableau)):
                    print(tableau[l][i])
                    o=0
        return unbounded
    
    def __searchPivotRow(self, tableau, pivotColumn):
        pivotRow=-1
        quotient=float("inf")
        for i in range(len(tableau)-1):
            if tableau[i][pivotColumn]!=0:
                memo=tableau[i][-1]/tableau[i][pivotColumn]
                if (tableau[i][pivotColumn] > 0) and (memo<quotient):
                    pivotRow=i
                    quotient=memo
        return [pivotRow,quotient]
    
    def __simplexStep(self,simplextableau,pivotColumn,pivotRow):
        pivotElement=simplextableau[pivotRow][pivotColumn]
        for i in range(len(simplextableau)):
            if i!=pivotRow:
                curElement=simplextableau[i][pivotColumn]
                factor=curElement/pivotElement
                #for k in range(simplextableau[i].shape[0]):
                simplextableau[i]=round(simplextableau[i]-factor*simplextableau[pivotRow],self.roundDigits)
                    #if (abs(simplextableau[i][k]) < self.tolerance):
                    #    simplextableau[i][k]=0
                        
        #for k in range(simplextableau[pivotRow].shape[0]):                            
        simplextableau[pivotRow]=round(simplextableau[pivotRow]/simplextableau[pivotRow][pivotColumn],self.roundDigits)  
            #if abs(simplextableau[pivotRow][k]) < self.tolerance:
            #    simplextableau[pivotRow][k]=0
        return simplextableau
    
    def __getSolution(self,tableau,n):
        center=zeros(n)
        radius=tableau[-1][-1]
        for column in range(2,(2+2*n)):
            i=self.__getSolutionRow(tableau,column,n)
            #print(i)
            if  i!=None and (column-2)%2==0:
                center[(column-2)/2]+=tableau[i][-1]
            elif i!=None:
                center[(column-2)/2]-=tableau[i][-1]
            else:
                print("not good")
        #self.printSol(center)
        return [center,radius]
    
    def __getSolutionRow(self,tableau,column,n):
        row=None
        foundOne=False;
        for i in range(len(tableau)-1):
            if tableau[i][column]==1 and not foundOne:
                foundOne=True
                row=i
            elif not abs(tableau[i][column])<self.tolerance:
                row=None
        
        return row            
    
    def printTableau(self,tableau):
        stderr.write("\n")
        for i in range(len(tableau)):
            stderr.write("\n")
            for j in range(tableau[i].shape[0]):
                stderr.write("%d "%tableau[i][j])
        stderr.write("\n")
    
    def printSol(self,center):
        for i in range(center.shape[0]):
            stderr.write("%f "%center[i])
        stderr.write("\n")
#asa=AdjustedSimplexAlg()
#challenges=[]
#responses=[]
#challenges.append(array([1,1,1,1,1]))
#responses.append(1)
#challenges.append(array([-1,1,1,1,1]))
#responses.append(-1)
#challenges.append(array([-1,1,1,1,-1]))
#responses.append(-1)
#challenges.append(array([1,-1,-1,1,1]))
#responses.append(1)
#challenges.append(array([-1,-1,1,1,-1]))
#responses.append(-1)

#challenges.append(array([1,1]))
#responses.append(-1)
#challenges.append(array([-1,1]))
#responses.append(1)
#[center,radius]=asa.solve(challenges,responses)
#for i in range(center.size):
#    print(center[i])