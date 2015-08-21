import random
import os
import sys
import math
 


inpath = "D://python2.7.6//MachineLearning//titanic-randomforest//data"
outfile1 = "D://python2.7.6//MachineLearning//titanic-randomforest//1.txt"
outfile2 = "D://python2.7.6//MachineLearning//titanic-randomforest//2.txt"
outfile3 = "D://python2.7.6//MachineLearning//titanic-randomforest//3.txt"
outfile4 = "D://python2.7.6//MachineLearning//titanic-randomforest//4.txt"
outfile5 = "D://python2.7.6//MachineLearning//titanic-randomforest//5.txt"
outfile6 = "D://python2.7.6//MachineLearning//titanic-randomforest//6.txt"
outfile7 = "D://python2.7.6//MachineLearning//titanic-randomforest//7.txt"
     

numCenter=30 #must not >=numPsg xi
######################

def loadData():
    global dataDic;dataDic={};global numPsg;global psgList;psgList=[];global featDic;
    
    for filename in os.listdir(inpath):
        content=open(inpath+'/'+filename,'r')
        data=content.readlines();
        feat=data[0].strip('\n')
        dataDic[feat]={};#print data[0].strip('\n')
        i=0
        for d in data[1:]:
            dataDic[feat][i]=d.strip('\n')
            if len(d.strip('\n'))<1:dataDic[feat][i]=-10
            
            i+=1
    #####################
    outPutfile=open(outfile1,'w')
    for feat,data in dataDic.items():
        #print len(data),feat,'loaded'
        numPsg=len(data)
        outPutfile.write(str(feat));
        outPutfile.write(str(data))
        outPutfile.write('\n')
    outPutfile.close()

    ########already get original data, now amend
    featDic={}
    for feat in dataDic.keys():
        featDic[feat]=set(dataDic[feat].values())
    del featDic['Survived']

    outPutfile=open(outfile2,'w')
    for k,v in featDic.items():
        outPutfile.write(str(k))
        outPutfile.write(str(v));
        outPutfile.write('\n')
    outPutfile.close()
    ############## str into float   
    for i in range(numPsg):
        eachPsg=[{},'tlable','plable']
        for feat in dataDic.keys():
            if feat in ['Pclass','Embarked','Sex','Cabin']:
                eachPsg[0][feat]=dataDic[feat][i]  ###i start from 0,
            elif feat in ['Age','Fare','Parch','SibSp']:
                eachPsg[0][feat]=float(dataDic[feat][i])
        eachPsg[1]=int(dataDic['Survived'][i])
        #########for one passenger,all feat finish
        psgList.append(eachPsg)

    
    ############################
    outPutfile=open(outfile3,'w')
    for psg in psgList:
        outPutfile.write(str(psg));
        outPutfile.write('\n')
    outPutfile.close()

def absentData():
    global dataDic;global numPsg;global psgList;global featDic;
    #####age absent:after study distribution between fare and age, random age at(15,50)
    for psg in psgList:
        if psg[0]['Age']==-10:
            if psg[0]['Fare']<=50:
                psg[0]['Age']=random.randint(0,80)
            else:psg[0]['Age']=random.randint(15,50)
    #####embark absent:after see distribution, find 1 'S' is dominant
    for psg in psgList:
        if psg[0]['Embarked']==-10:psg[0]['Embarked']='S'
    ###########

    ###########
    outPutfile=open(outfile4,'w')
    for psg in psgList:
        outPutfile.write(str(psg));
        outPutfile.write('\n')
    outPutfile.close()

def transformFeat():#all include continuous variabal into 0 1 
    global dataDic;global numPsg;global psgList;global featDic;
    ###############str feat into 0 1
    global strFeatDic;
    strFeatDic={'Pclass':('1','2','3'),\
                'Embarked':('C','S','Q'),\
                'Sex':('female','male'),\
                'Cabin':('C','A','B','D','E','F','G','T')}
    for psg in psgList[:]:
        for f,v in strFeatDic.items():
            for i in v[1:]:
                if psg[0][f]==i:
                    psg[0][i]=1
                else:psg[0][i]=0
        ########cabin str 'b12' into(zimu,shuzi) 
        cab=psg[0]['Cabin'] #cab may =(zimu,shuzi)-10 DATA ABSENT or DATA PRESENT str
        if type(cab)==str and cab!=-10: #if 'D' not  'D12'
            zimu,shuzi=splitCab(cab)
            psg[0]['Cabin']=(zimu,shuzi);#print psg[0]['Cabin']
        if cab==-10:
            psg[0]['Cabin']=(-10,-10)
            
        
        '''############### now all cabin =(zimu,shuzi)
        cab=psg[0]['Cabin'] #cab only =(zimu,shuzi)
        for i in ('A','B','C','D','E','F','G','T')[1:]:
            if i in cab:
                psg[0][i]=1
        psg[0]['cabNum']=cab[1]'''
        ##############del str feat
        del psg[0]['Pclass'];del psg[0]['Embarked'];del psg[0]['Sex'];#del psg[0]['Cabin']
         

    ############now continuous into 0-1
    continuFeat={'Fare':{'fare1':(0,100),'fare2':(101,200),'fare3':(201,300)},\
                 
                 'Age':{'age1':(0,10),'age2':(11,20),'age3':(21,30),'age4':(31,40),'age5':(41,50),'age6':(51,100)},\
                 
                 'Parch':{'child1':(0,1),'child2':(2,3),'child3':(4,5)},\
                 
                 'SibSp':{'sib1':(0,1),'sib2':(2,3)}}
    
    
    for psg in psgList[:]:
        for k,v in continuFeat.items():
            for f,num in continuFeat[k].items():
                psg[0][f]=0
                if int(psg[0][k]) in range(num[0],num[1]+1):
                    psg[0][f]=1

        #################new structure for psg:[{feat:0,1 include survived},truelabel 01, predict,{center and distance}]
        psg[0]['Survived']=psg[1]
        psg[1]=[psg[0]['Cabin'][0],psg[0]['Cabin'][1]];#(zimu,shuzi)(-10,-10)
        ##############
        del psg[0]['Fare'];del psg[0]['Age'];del psg[0]['Parch'];del psg[0]['Cabin']
        del psg[0]['SibSp'];

        
        

    ##########################    
    outPutfile=open(outfile5,'w')
    for psg in psgList:
        outPutfile.write(str(psg));
        outPutfile.write('\n')
    outPutfile.close()    
'''one psg:[{'child1': 1, 'child3': 0, 'sib1': 1,
'3': 1, '2': 0, 'Survived': 0, 'age5': 0, 'A': 0,
'sib2': 0, 'child2': 0, 'B': 0, 'E': 0, 'D': 0,
'G': 0, 'F': 0, 'age6': 0, 'age4': 0, 'age3': 1,
'age2': 0, 'age1': 0, 'Q': 0,'S': 1, 'T': 0,
'fare2': 0, 'fare1': 1, 'fare3': 0, 'male': 1}, (-10,-10)or ('C', 85), 'plable']'''

def initialCenter():
    global psgList;global center
    feaD=psgList[0][0].keys()
    center={}
    for i in range(numCenter):
        center[i]={}
        for fea in feaD:
            center[i][fea]=random.sample([0,1],1)[0]
    #print center

def cluster():
    global psgList;global center;global numPsg
    #############calc distance
    for psg in psgList[:]:
        ###for each psg find closest center and distance
        minDis=None;minCen=0;
        for cen,fea in center.items():
            dis=calcDis(fea,psg[0])
            if minDis ==None or minDis>dis:
                minDis=dis
                minCen=cen
        psg[2]=[minCen,minDis]
        #print psg[2]
    ################calc total distance
    totalLose=0.0
    for psg in psgList[:]:
        totalLose+=psg[2][1]
    print '%d centerios ,total psg distance %f'%(numCenter,totalLose)
    ###############
    #for psg in psgList[:7]:
        #print psg[1],psg[2]
    ##############calc new centerior
    feaD=psgList[0][0].keys()
    for cen,fea in center.items():
        for f,v in fea.items():
            ##########calc mean by each center each feat  
            meani=0.0;numMember=0.0
            for psg in psgList[:]:
                if psg[2][0]==cen:
                    meani+=psg[0][f]
                    numMember+=1
            meani/=(numMember+0.0001)
            center[cen][f]=meani
    #print center
    return totalLose
            
        
    

def show():
    global psgList
    ###############show distribution,see each centerior/cluster, what kind of zimu shuzi in it
    global zimuL;global shuziL;
    zimuL=[];shuziL=[]
    for i in range(numCenter):
        eachCzm=[];eachCsz=[];
        for psg in psgList[:]:
            if psg[2][0]==i:
                eachCzm.append(psg[1][0])
                eachCsz.append(psg[1][1])
        zimuL.append(eachCzm)
        shuziL.append(eachCsz)

    #########
    for i in range(len(zimuL)):
        zimuL[i]=set(zimuL[i])
    for i in range(len(shuziL)):
        shuziL[i]=set(shuziL[i])
    #print 'finally','zimu',zimuL,'shuzi',shuziL 

def labelCabin():
    global psgList;global zimuL;global shuziL; 
    #############remove -10 from set
    for i in range(len(zimuL)):
        if -10 in zimuL[i]:
            zimuL[i].remove(-10)
            ######what if after remove 0 in the set , some cluster is empty itself before remove -10
            if len(zimuL[i])==0:
                zimuL[i]=['C','A','B','D','E','F','G','T']
     
    for i in range(len(shuziL)):
        if -10 in shuziL[i]:
            shuziL[i].remove(-10)
            if len(shuziL[i])==0:
                shuziL[i]=[random.randint(1,150)]
    #print '1',shuziL,zimuL
    ############label (cabin zimu , cabin shuzi)
    for psg in psgList:
        
        ###########label cabin zimu
        if psg[1][0]==-10:
            for i in range(numCenter):
                if psg[2][0]==i:
                     psg[1][0]=random.sample(zimuL[i],1)[0]  #sample return [0] not 0
        ###########label cabin shuzi
        if psg[1][1]==-10:
            for i in range(numCenter):
                if psg[2][0]==i:
                     psg[1][1]=random.sample(shuziL[i],1)[0]  #sample return [0] not 0
        ######
        
    ####################
    outPutfile=open(outfile6,'w')
    for psg in psgList:
        outPutfile.write(str(psg));
        outPutfile.write('\n')
    outPutfile.close()
    ##################continuous variable cabin number into 01
    numCab={'cabN1':(0,10),'cabN2':(11,20),'cabN3':(21,30),'cabN4':(31,40),\
                 'cabN5':(41,50),'cabN6':(51,60),'cabN7':(61,70),'cabN8':(71,80),\
                 'cabN9':(81,90),'cabN10':(91,100),'cabN11':(101,110),'cabN12':(111,120),\
                 'cabN13':(121,130)}
    for psg in psgList:
        labelCab=psg[1][0]
        if labelCab in psg[0]:
            psg[0][labelCab]=1
        ##########
        for k,v in numCab.items():
            psg[0][k]=0
            if int(psg[1][1]) in range(int(v[0]),int(v[1])+1):
                psg[0][k]=1
        ###########
        psg[1]=psg[0]['Survived']
        del psg[0]['Survived']
    ################################
    outPutfile=open(outfile7,'w')
    for psg in psgList:
        for k,v in psg[0].items():
            outPutfile.write(str(k))
            outPutfile.write(':')
            outPutfile.write(str(v))
            outPutfile.write(' ')
        outPutfile.write(str(psg[1]))
        outPutfile.write('\n')
    outPutfile.close()
        
        
        
    
            


            

################################# SUPPORT               
def splitCab(cab):
    
    cabL=[];zimu=-10;shuzi=-10
    if ' ' in cab:
        cabL=cab.split(' ');#print cabL
        for pos in cabL:
            if len(pos)==1:
                zimu=pos
                #print zimu
            if len(pos)>1:
                zimu=pos[0];shuzi=int(pos[1:])
                #print zimu,shuzi
                continue  ##important choose long length 'a123'rather than 'a'
    if ' 'not in cab:
        if len(cab)==1:
            zimu=cab
        if len(cab)>1:
            zimu=cab[0];shuzi=int(cab[1:])
            
    return zimu,shuzi
            
def calcDis(dic1,dic2):
    dis=0.0
    for k,v in dic1.items():
        dis+=(v-dic2[k])**2
    dis=math.sqrt(dis)
    return dis
        
    
        
 

###########################main
loadData()


absentData()
transformFeat()
######to cluter in order to see distribution of cabin group
initialCenter()
lose1=cluster()

lose0=0.0
i=0
while abs(lose0-lose1)>0.01 and i<30:
    lose0=lose1
    lose1=cluster()
    i+=1
    
show()

###################
labelCabin()


#3 centerios ,total psg distance 1170.226266
#100 centerios ,total psg distance 714.829224
#8 centerios ,total psg distance 1040.752066
#700 centerios ,total psg distance 630.691374 ,without age 413
#800 centerios ,total psg distance 608.408353
