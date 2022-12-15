import pandas as pd
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA#导入ARIMA模型
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf#在graphics下面有一个专门画ACF和PACF的图片
from matplotlib.font_manager import FontProperties
myfont=FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')

stock=pd.read_csv('data_stock1.csv',delimiter=',',index_col=0,parse_dates=[0],encoding='GBK')
#注意其中的parse_dates是将第一列作为索引列，按照正常情况基本是将2022-09-24（标准格式）这样的格式用to_datetime将其转化为电脑可识别的形式
stock.isnull().sum()
#确定该数据无缺失值
'''
一般处理数据有高力度转化和低力度转化
低力度用to_timestamp()会映射到时间的第一天
高力度转换用to_period()会映射到时间的第一天
'''
# =============================================================================
# resample的方法处理数据的时间频率，可以分为日，周，月，季，年
# =============================================================================
stock_week=stock.收盘价.resample('W-MON').mean()#W-MON的意思是week中的monday

#选择训练集和测试机
train_number=stock_week.shape[0]*0.90
stock_train=stock_week.iloc[:780]#训练集占比90%
stock_test=stock_week.iloc[780:]#测试集占比10%


#画出训练集图像
plt.figure(figsize=(12,6),dpi=600)
plt.plot(stock_train)
plt.xlabel('时间',font=myfont,fontsize=15)
plt.ylabel('股票收盘价',font=myfont,fontsize=15)
plt.title('股票收盘价——训练集',font=myfont,fontsize=20)
plt.show()

#发现图片是属于有截距、有趋势的，我们运用adfuller中的ct及constant和trend来进行单位根检验
from statsmodels.tsa.stattools import adfuller
p=adfuller(stock_train,regression='ct')
print(p[1])
#有截距、有趋势下接受单位根为0的原假设，说明有单位根，该序列为随机性非平稳，此时用差分解决

stock_train_diff=stock_train.diff()
plt.figure(figsize=(12,8),dpi=600)
stock_train_diff.dropna(inplace=True)#删除一阶差分后的空缺值
stock_train_diff.plot()#发现改图为无截距，且无趋势
plt.title('差分后的训练集',font=myfont,fontsize=20)
plt.xlabel('时间',font=myfont,fontsize=15)
plt.ylabel('股票收盘价',font=myfont,fontsize=15)
plt.show()
plt.rcParams['axes.unicode_minus']=False
p_diff=adfuller(stock_train_diff,regression='nc')
print(p_diff[1])
#此时拒绝原假设，即该序列平稳了

#%%
#进行随机性检验（非白噪声），画出ACF图
plot_acf(stock_train_diff)#该图中当期值与滞后一期的值是有相关性的，即该序列有相关性，不是白噪声，
#滞后期在1处截尾

#PACF检验
plot_pacf(stock_train_diff)#滞后期在1处截尾

#识别定阶，建立ARIMA模型
model=ARIMA(stock_train,order=(1,1,1)).fit()
stock_test_predict=model.predict('20141215','20160801',dynamic=True,typ='levels')#level在这里的意思是还原差分后的数据

#画图比较
plt.figure(figsize=(16,8),dpi=600)
plt.rcParams['font.sans-serif']=['simhei']
plt.plot(stock_test_predict,linewidth=3,c='red',label='测试集预测')
plt.plot(stock_week,linewidth=2,c='blue',label='原始数据')
plt.xlabel('时间',font=myfont)
plt.ylabel('股票价格',font=myfont)
plt.title('预测',font=myfont)
plt.legend(fontsize=20)

#%%
#最终进行实际预测
stock_predict=model.predict('20141215','20171218',dynamic=True,typ='levels')
stock_predict=stock_predict['20160808':]

plt.figure(figsize=(16,8),dpi=600)AA
plt.rcParams['font.sans-serif']=['simhei']
plt.plot(stock_test_predict,linewidth=3,c='red',label='测试集预测')
plt.plot(stock_week,linewidth=2,c='blue',label='原始数据')
plt.plot(stock_predict,linewidth=4,c='orange',label='预测数据')
plt.xlabel('时间',font=myfont)
plt.ylabel('股票价格',font=myfont)
plt.title('预测',font=myfont)
plt.legend(fontsize=20)


#%%
#第二种方法识别定阶是取最小AIC，对比ACF、PACF
import statsmodels.api as sm
model_aic=sm.tsa.arma_order_select_ic(stock_train,max_ar=5,max_ma=5,ic=['aic'])
print(f'最好的p和q为：{model_aic.aic_min_order}') 
#结果为p=3,q=3

model_aic=ARIMA(stock_train,order=(3,1,3)).fit()
stock_test_aic=model_aic.predict('20141215','20160801',dynamic=True,typ='levels')

from sklearn.metrics import mean_squared_error

MSE1=mean_squared_error(stock_test,stock_test_predict)#通过ACF,PACF确定的
MSE2=mean_squared_error(stock_test, stock_test_aic)#通过AIC最小确定的
print(f'通过ACF和PACF确定阶数的MSE为{MSE1}\n通过AIC确定阶数的MSE为{MSE2}')
#最后比较为MSE1小，p=1，q=1


'''
当序列是拒绝原假设时，就是一个确定性趋势非平稳，用生成残差解决
1.通过date_range()生成一个新的index，然后进行以t为自变量的一元线性回归，最终生成估计的y^，error=y-y^
2.用error来进行平稳性检验、随机性检验，最终用error来预测序列

运用.diff()函数，1.2阶差分指的是diff().diff()。 2. q次一阶差分指的是diff(q)主要解决序列具有周期性的问题
'''









