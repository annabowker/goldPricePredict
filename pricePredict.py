"""
This program predicts the price of an ounce of gold and the NASDAQ index using 2 different methods.
It uses MPI4py to speed up some of the code.

Author: Anna Bowker
"""

#run with: mpiexec -np 2 python project.py
from mpi4py import MPI
import numpy
from datetime import datetime
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression
import quandl as ql


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()


#************GET HISTORICAL GOLD PRICES****************
#get arrays
start = dt.datetime(1970,1,1)
end = dt.date.today()

quandlKey = "your quandl key"
ql.ApiConfig.api_key = quandlKey

#gets closing value of price of an oz of gold in USD
gold = ql.get('LBMA/GOLD.2',start_date = start,end_date = end,returns = 'numpy')

date = []
value = []
for i in range(len(gold)):
    date.append(gold[i][0])
    value.append(gold[i][1])

#get most recent value
import requests
from bs4 import BeautifulSoup
data = requests.get('https://www.jmbullion.com/charts/gold-price/')
soup = BeautifulSoup(data.text, 'html.parser')
currentPrice = soup.find_all('div', {'id':'gounce'})
mostRecentPrice = currentPrice[0].text
currentDate = soup.find_all('div', {'id':'spot_time'})
value.append(float(mostRecentPrice))
date.append(dt.datetime.now())

if value[-1] == value[-2] or value[-1] == 0:
    value.pop(-1)
    date.pop(-1)


#*************GET NASDAQ VALUES*****************
import re
from io import StringIO
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np

from requests.exceptions import HTTPError


class YahooFinanceHistory:
    timeout = 2
    crumb_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
    crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
    quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history&crumb={crumb}'

    def __init__(self, symbol, days_back=7):
        self.symbol = symbol
        self.session = requests.Session()
        self.dt = timedelta(days=days_back)

    def get_crumb(self):
        response = self.session.get(self.crumb_link.format(self.symbol), timeout=self.timeout)
        response.raise_for_status()
        match = re.search(self.crumble_regex, response.text)
        if not match:
            raise ValueError('Could not get crumb from Yahoo Finance')
        else:
            self.crumb = match.group(1)

    def get_quote(self):
        if not hasattr(self, 'crumb') or len(self.session.cookies) == 0:
            self.get_crumb()
        now = datetime.utcnow()
        dateto = int(now.timestamp())
        datefrom = int((now - self.dt).timestamp())
        url = self.quote_link.format(quote=self.symbol, dfrom=datefrom, dto=dateto, crumb=self.crumb)
        response = self.session.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), parse_dates=['Date'])

if rank == 0:
	trying=True
	while trying:
		trying=False
		try:
			#NADAQ 100
			#df = YahooFinanceHistory('^NDX', days_back=12500).get_quote()

			#NASDAQ Composite
			#df = YahooFinanceHistory('^IXIC', days_back=12500).get_quote()
			df = YahooFinanceHistory('^IXIC', days_back=17600).get_quote()

			#Dow Jones Industrial
			#df = YahooFinanceHistory('^DJI', days_back=12500).get_quote()

			#closeData = np.array(df['Adj Close'])
			closeData = df['Adj Close'].tolist()
		except HTTPError:
			trying=True
else:
	closeData = None

closeData = comm.bcast(closeData, root=0)



def correlate(series,temp):
    corrArray = [] #array of correlated values at each shift
    n = len(temp)

    delta = len(series)-len(temp)
    for i in range(delta+1): 
        shiftedSeries = [] #array of n values in order that shifts through the series of all data
        for j in range(i,len(temp)+i,1):
            shiftedSeries.append(series[j])
        #print(shiftedSeries)
        #turn into data frame, correlate, and add to array of correlation values
        d = {'a':temp,'b':shiftedSeries}
        df = pd.DataFrame(data = d)
        c = pd.Series.corr(df['a'],df['b'])

        corrArray.append(c)
        
    return corrArray

#perform operations then communicate
def communicate(operation,recvdata,array2):
  if rank == 0:
    #perform operations
    recvdata = operation(recvdata,array2)

    #get all data from workers and concatenate array into one final array
    for i in range(1,size):
      data = numpy.empty_like(recvdata, dtype=numpy.float64)
      data = comm.recv(source=i)
      recvdata = numpy.concatenate((recvdata,data), axis = None)

    #return recvdata1,recvdata2
    return recvdata

  else:
    #perform operations
    recvdata = operation(recvdata,array2)
    #send data to master
    comm.send(recvdata, dest=0)

    #return recvdata[0],recvdata[1]
    return recvdata

  #return recvdata1


def parallel(operation,array,array2):
    n = len(array)
    remainder = n%size

    
    cutoff = numpy.empty(size,dtype=int) #cutoff values, where you want array to be cut
    cutoff[size-1] = int(n/size) + remainder #make last array take care of remainder when data can't be split evenly
    for i in range(size-1):
      cutoff[i] = int(n/size) + len(array2)-1
    counts = tuple(cutoff)
    #print(counts)

    cutoff2 = numpy.empty(size,dtype=int) #displacement values, where you want sub-array at each rank to start
    cutoff2[0] = 0 #make first sub array start at 0
    for i in range(1, size):
      #cutoff2[i] = int(cutoff[i-1] + cutoff2[i-1])
      cutoff2[i] = i*int(n/size)
    dspls = tuple(cutoff2)
    #print(dspls)

    senddata = array

    recvdata = numpy.empty(cutoff[rank],dtype=numpy.float64)
    comm.Scatterv([senddata,counts,dspls,MPI.DOUBLE],recvdata,root=0)
    #print( 'on task',rank,'after Scatterv:    data = ',recvdata)

    #OPERATIONS AND THEN COMMUNICATE
    recvdata1 = communicate(operation,recvdata,array2)
    return recvdata1



def method1(values, name):
  #factor series as all points minus last year, temp as the last # to correlate
  splitArray = np.split(values,[-500])
  series = splitArray[0] #array of values before last 500 days
  comp = splitArray[1]  #array of the last 500 days values

  startTime = datetime.now()
  #run with parallel processing
  a=parallel(correlate,series,comp)

  time = datetime.now() - startTime
  print(name + 'rank: ' + str(rank) + ', time:' + str(time))

  b=np.argmax(a)

  if(rank == 0):
    maxCorrIndex = b
    last = len(comp)
    delta = float(series[maxCorrIndex + last + 1] - series[maxCorrIndex + last])
    prediction1 = float(comp[-1] + delta)
    delta = series[maxCorrIndex + last + 2] - series[maxCorrIndex + last + 1]
    prediction2 = prediction1 + delta
    delta = series[maxCorrIndex + last + 3] - series[maxCorrIndex + last+ 2]
    prediction3 = prediction2 + delta
    pred = [prediction1, prediction2,prediction3]
    #print(pred)
    return pred

def method2(value):
    #date.append(dt.date.today() + dt.timedelta(days=1))
    value.append(0)

    #d = {'Date': date, 'Close': value}
    d = {'Close': value}
    df = pd.DataFrame(data = d)

    #grab mean of previous 3, 5, 9, 12 day window shift one and do same until column is full
    df['S_3'] = df['Close'].shift(1).rolling(window=3).mean()
    df['S_5'] = df['Close'].shift(1).rolling(window=5).mean()
    df['S_9'] = df['Close'].shift(1).rolling(window=9).mean()
    df['S_12'] = df['Close'].shift(1).rolling(window=12).mean()
    #grab above means for "tomorrow"
    s3 = df.loc[len(df)-1,'S_3']
    s5 = df.loc[len(df)-1,'S_5']
    s9 = df.loc[len(df)-1,'S_9']
    s12 = df.loc[len(df)-1,'S_12']

    df = df.drop([len(df)-1]) #drop the "tomorrow" row
    df = df.dropna() #get rid of NaN rows

    x = df[['S_3','S_5','S_9','S_12']]
    y = df['Close']

    xTest = pd.DataFrame(data ={'a':[s3],'b':[s5],'c':[s9],'d':[s12]})
    linear = LinearRegression().fit(x,y)
    m3 = round(linear.coef_[0],2)
    m5 = round(linear.coef_[1],2)
    m9 = round(linear.coef_[2],2)
    m12 = round(linear.coef_[3],2)
    c = round(linear.intercept_,2)
    #predict = m3*s3 + m5*s5 + m9*s9 + m12*s12 + c
    #predict = round(predict,2)
    predictedPrice = linear.predict(xTest)
    predictedPrice = round(predictedPrice[0],2)
    #prediction = [predict,predictedPrice]
    return predictedPrice;


#*************RUN*******************
g = method1(value, 'Gold: ')
n = method1(closeData, 'NASDAQ: ')

if rank==0:
  print("\nCurrent Gold Price:   $" + str(value[-1]))
  print("Current NASDAQ Index: " + str('{:.2f}'.format(closeData[-1])))

  f = method2(value)
  h = method2(closeData)
  print("\nLinear Regression Prediction:")
  print("Predicted Gold Price: \n\t" + str(dt.date.today() + dt.timedelta(days=1)) + ":\t$" + str(f))
  print("Predicted NASDAQ Index: \n\t" + str(dt.date.today() + dt.timedelta(days=1)) + ":\t" + str('{:.1f}'.format(h)))

  print("\nCorrelation Prediction:")
  print("Predicted Gold Price: \n\t" + str(dt.date.today() + dt.timedelta(days=1)) + ":\t$" + str('{:.2f}'.format(g[0])))
  print("\t" + str(dt.date.today() + dt.timedelta(days=2)) + ":\t$" + str('{:.2f}'.format(g[1])))
  print("\t" + str(dt.date.today() + dt.timedelta(days=3)) + ":\t$" + str('{:.2f}'.format(g[2])))
  print("Predicted NASDAQ Index: \n\t" + str(dt.date.today() + dt.timedelta(days=1)) + ":\t" + str('{:.1f}'.format(n[0])))
  print("\t" + str(dt.date.today() + dt.timedelta(days=2)) + ":\t" + str('{:.1f}'.format(n[1])))
  print("\t" + str(dt.date.today() + dt.timedelta(days=3)) + ":\t" + str('{:.1f}'.format(n[2])))