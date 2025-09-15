import yfinance as yf
# 输入
symbol = 'AMD'
start = '2014-01-01'
end = '2018-08-27'

dataset=yf.pdr_override(symbol,start,end)
dataset.head()