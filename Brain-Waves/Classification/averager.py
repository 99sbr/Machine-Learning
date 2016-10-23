import pandas as pd 
pred1=pd.read_csv("pred1.txt")
pred2=pd.read_csv("pred2.txt")
frame=pd.DataFrame(data=pred1.y,columns=['y'])
	for i in range(0,len(pred1)):
	    if (pred1.y[i]==1) & (pred2.y[i]==1):
	        frame.y[i]=1
	    elif (pred1.y[i]==-1) & (pred2.y[i]==-1):
	        frame.y[i]=-1
	    elif (pred1.y[i]==-1) & (pred2.y[i]==1):
	        frame.y[i]=1
	    elif (pred1.y[i]==1) & (pred2.y[i]==-1):
	        frame.y[i]=1
file = open("final.txt", "a")
file.write("Time,Y\n")
for i in range(0,len(pred1)):
    file.write(str(3000+i))
    file.write(",")
    file.write(str(frame.y[i]))
    file.write("\n")
file.close()