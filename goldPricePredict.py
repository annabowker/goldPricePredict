import pandas as pd
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression
import quandl as ql
import tkinter

start = dt.datetime(1970,1,1)
end = dt.date.today()
#end = dt.datetime(2018,7,24)

quandlKey = "PDszhZvvdzBEcgW7DXby"
ql.ApiConfig.api_key = quandlKey

#gets closing value of price of an oz of gold in USD
gold = ql.get('LBMA/GOLD.2',start_date = start,end_date = end,returns = 'numpy')
#gold2 = ql.get('WGC/GOLD_DAILY_USD',start_date = start,end_date = end, retruns = 'numpy')

date = []
value = []
for i in range(len(gold)):
    date.append(gold[i][0])
    value.append(gold[i][1])
#last = value[-1]
#day = str(date[-1])
#day = day.split('T')
#day = day[0]

#textbox
window = tkinter.Tk()
window.config(height=300,width=300)
window.title("Gold Price Predictor")
window.wm_iconbitmap('heart.icns')
photo = tkinter.PhotoImage(file = "gold2.gif")

w = tkinter.Label(window,image=photo)
#w.place(x=0,y=0,relwidth=1,relheight=1)
w.place(relx=0.5,rely=0.5,anchor='center')
label = tkinter.Label(window,text = "Use this to predict \ntomorrow's price of \none oz of gold", bg = 'darkgoldenrod')
label.place(relx=0.5,rely=0.4,anchor='center')

window.button = tkinter.Button(window, text = 'Run', command = window.destroy)
window.button.place(relx=.5,rely=.6,anchor='center')

window.mainloop()

def closeWin():
    value.append(float(win.entry.get()))
    date.append(dt.datetime.now())
    win.destroy()

win = tkinter.Tk()
win.config(height=240,width=300)
win.title("Gold Price Predictor")
win.wm_iconbitmap('heart.icns')
photo2 = tkinter.PhotoImage(file = "gold.gif")
w2 = tkinter.Label(win,image=photo2)
w2.place(x=0,y=0,relwidth=1,relheight=1)
label = tkinter.Label(win,text = "Enter the most recent price of gold (in USD):\nIf unknown, enter zero",bg = 'darkgoldenrod3')
label.place(relx=0.5,rely=0.4,anchor='center')
win.entry = tkinter.Entry(win)
win.entry.place(relx=.5,rely=.55,anchor='center')
win.button = tkinter.Button(win, text = 'OK', command = closeWin)
win.button.place(relx=.5,rely=.65,anchor='center')

win.mainloop()

if value[-1] == value[-2] or value[-1] == 0:
    value.pop(-1)
    date.pop(-1)

last = value[-1]
day = str(date[-1])
day = day.split('T')
day = day[0]
day = day.split(' ')
day = day[0]

#will take the most recent n entries and compare with whole set and predict next
#n = how long of a series to compare(must be at least 2)
def method3(series, n):
    array = [] #array of correlated values at each shift
    temp = [] #array of the most recent n values in order
    maximumCorrelation = 0
    maxCorrIndex = 0
    for i in range(len(series)):
        if i>=(len(series)-n):
            temp.append(series[i])
    for i in range(len(series)-n+1-500): #minus 500 to ignore the most recent two years in comparison
        shiftedSeries = [] #array of n values in order that shifts through the series of all data
        for j in range(i,len(temp)+i,1):
            shiftedSeries.append(series[j])
        #print(shiftedSeries)
        #turn into data frame, correlate, and add to 'array'
        d = {'a':temp,'b':shiftedSeries}
        df = pd.DataFrame(data = d)
        c = pd.Series.corr(df['a'],df['b'])
        if c > maximumCorrelation:
            maximumCorrelation = c
            maxCorrIndex = i
        array.append(c)

    delta = float(series[maxCorrIndex + n + 1] - series[maxCorrIndex + n])
    prediction1 = float(series[-1] + delta)
    delta = series[maxCorrIndex + n + 2] - series[maxCorrIndex + n + 1]
    prediction2 = prediction1 + delta
    delta = series[maxCorrIndex + n + 3] - series[maxCorrIndex + n+ 2]
    prediction3 = prediction2 + delta
    pred = [prediction1, prediction2,prediction3]
    return pred;

def method2n1(date,value):
    date.append(dt.date.today() + dt.timedelta(days=1))
    value.append(0)

    d = {'Date': date, 'Close': value}
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
    predict = m3*s3 + m5*s5 + m9*s9 + m12*s12 + c
    predict = round(predict,2)
    predictedPrice = linear.predict(xTest)
    predictedPrice = round(predictedPrice[0],2)
    prediction = [predict,predictedPrice]
    return prediction;
    
g=method3(value,4000)
f=method2n1(date,value)
#print("Most recent price per oz. on " + str(day) + ": $" + str('{:.{prec}f}'.format(last, prec = 2)))
#print("Tomorrow's Predicted Gold Price (Method 1): $" + str(f[0]))
#print("Tomorrow's Predicted Gold Price (Method 2): $" + str(f[1]))
#print("Tomorrow's Predicted Gold Price (Method 3): $" + str('{:.{prec}f}'.format(g[0],prec=2)))

#textbox
window2 = tkinter.Tk()
window2.config(height=300,width=400)
window2.title("Predictions")
window2.wm_iconbitmap('heart.icns')
photo3 = tkinter.PhotoImage(file = "gold3.gif")
w3 = tkinter.Label(window2,image=photo3)
#w3.place(x=0,y=0,relwidth=1,relheight=1,anchor='center')
w3.place(relx=0.5,rely=0.5,anchor='center')
t = "Most recent price per oz. on " + str(day) + ": $" + str('{:.{prec}f}'.format(last, prec = 2)) + "\n\nTomorrow's Predicted Gold Price (Method 1): $" + str(f[0]) + "\nTomorrow's Predicted Gold Price (Method 2): $" + str(f[1]) + "\nTomorrow's Predicted Gold Price (Method 3): $" + str('{:.{prec}f}'.format(g[0],prec=2))
label = tkinter.Label(window2,text = t,bg = 'darkgoldenrod')
label.place(relx=0.5,rely=0.4,anchor='center')
window2.button = tkinter.Button(window2, text = 'Done', command = window2.destroy)
window2.button.place(relx=.5,rely=.8,anchor='center')

window.mainloop()

