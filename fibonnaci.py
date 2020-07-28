n=int(input())
f0=0
f1=1
l=[]
l.append(0)
l.append(1)
for i in range(2,n):
	f2=f0+f1
	l.append(f2)
	f0=f1
	f1=f2
print(*l,sep=" ,")
