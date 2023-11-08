from scipy.stats import truncnorm
import math
import random
import numpy as np
from sympy import limit, Symbol, oo, diff, integrate
import matplotlib.pyplot as plt

#函數設定
def test(n, p):
	summ = 0
	for i in range(218):
		ran = random.random()
		if( p >= ran):
			summ = summ + 1
	return summ
#參數設定
n = 200						#單次試驗次數
p = 0.012					#成功機率		
#重複試驗+作圖
xline = []
xdata = []
xdata2 = []
summ = 0
yline = range(n)
x = np.zeros(n,int)
for i in range(100000):
	num = test(n, p)
	x[num] = x[num] + 1
for i in range(20):
	xline.append(x[i]/100000)
	summ = summ + x[19-i]/100000
	xdata.append(summ)
for i in range(20):
	xdata2.append(xdata[19-i])
plt.bar(range(20),xline,color='b',label='probability of real test')
plt.plot(range(0,20),xdata2,color='b',marker='x',linestyle='None',label='cumulative probability of real test')
print(xdata2[0])


#函數建立
def pdf(sigma, mu, x):
	x2 = (x-mu)/sigma
	return (1/(math.sqrt(2*math.pi)*sigma))*math.exp(-0.5*(x2**2))

def tnd(sigma, mu, x, a, b):

	if(b==oo):
		return pdf(sigma, mu, x)/(1-scdf((a-mu)/sigma))
	elif(a==-oo):
		return pdf(sigma, mu, x)/scdf((b-mu)/sigma)
	else:
		return pdf(sigma, mu, x)/(scdf((b-mu)/sigma)-scdf((a-mu)/sigma))

def scdf(x):
	return 0.5*(1+math.erf(x/math.sqrt(2)))

#參數設定
p = 0.012					#機率
n = 200						#試驗次數

#導出參數
mu = p*n					#期望值
var = n*p*(1-p)				#變異數
sigma = math.sqrt(var) 		#標準差

#空陣列設定
xline_1 = []
xline_2 = []
xline_2_2 = []
xline_3 = []
xline_4 = []
xline_4_2 = []
yline_1 = []
yline_2 = []
sum_1 = 0
sum_2 = 0

#連續模型導出
for i in range(0, 30000):
	a=tnd(sigma, mu, 20*(i/30000), 0, oo)
	xline_1.append(a)
	yline_1.append(20*(i/30000))
for i in range(0,30000):
	sum_1=sum_1+xline_1[29999-i]*20/30000
	xline_2.append(sum_1)
for i in range(0,30000):
	xline_2_2.append(xline_2[29999-i])

for i in range(-30000, 30000):
	b=pdf(sigma, mu, 20*(i/30000))
	xline_3.append(b)
	yline_2.append(20*(i/30000))
for i in range(0,60000):
	sum_2=sum_2+xline_3[59999-i]*20/30000
	xline_4.append(sum_2)
for i in range(0,60000):
	xline_4_2.append(xline_4[59999-i])

#連續模型導出
plt.style.use('bmh')
plt.ylim(0,1.1)
plt.xlim(0,20)
plt.xlabel("1.2%,Number of wifu")
plt.ylabel("Probability")
plt.xticks(range(20))
plt.plot(yline_1,xline_1,color='r')		#tnd
plt.plot(yline_1,xline_2_2,color='r')
plt.plot(yline_2,xline_3,color='g')		#pdf
plt.plot(yline_2,xline_4_2,color='g')

#空陣列設定
xdata_1 = []
xdata_2 = []
xdata_3 = []
xdata_4 = []
ydata_1 = []
ydata_2 = []

#取點
for i in range(20):
	xdata_1.append(xline_1[int(30000*i/20)])
	xdata_2.append(xline_2_2[int(30000*i/20)])
	xdata_3.append(xline_3[int(30000+30000*i/20)])
	xdata_4.append(xline_4_2[int(30000+30000*i/20)])
ydata = range(20)

#取點導出+作圖
plt.bar(ydata,xdata_1,color='r',label='probability of tnd')								#tnd
plt.plot(ydata,xdata_2,color='r',marker='x',linestyle='None',label='cumulative probability of tnd')
plt.bar(ydata,xdata_3,color='g',linestyle='None',label='probability of pnd')				#pdf
plt.plot(ydata,xdata_4,color='g',marker='x',linestyle='None',label='cumulative probability of pdf')
plt.legend(loc='upper right')

for i in range(20):
	print("tnd下 恰抽到",i,"張的機率：",xdata_1[i])
for i in range(20):
	print("pdf下 恰抽到",i,"張的機率：",xdata_3[i])
for i in range(20):
	print("tnd下 至少抽到",i,"張的機率：",xdata_2[i])
for i in range(20):
	print("pdf下 至少抽到",i,"張的機率：",xdata_4[i])

plt.show()	
