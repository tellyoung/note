-----------------------------------------
data = pd.DataFrame(dict, columns=, index=)
index = pd.Index(['index_name', 'index_name'], name='id')
data.copy()
data.head() # data.tail()
data["col_name"] # 提取col_name列 data.col_name
data.rename(index=str.title, columns=str.upper) # data["col_name"].str.lower() / title首字母大写
data.rename(index={'col_name': 'new_name'}, inplace=True)
data.columns  
data.index.name # data.columns.name / data.index.names 多层索引
data['new_name'] = data.col_name == 'abc'
data[(np.abs(data) > 3).any(1)] # 选中但凡有一列大于3的所有行
data.take([index]) # 提取[index]中的行
data.sample(n=6) # 返回六行的随机子集
data.T
data.info()
data.index[1:] 	 # pd.Index(np.arange(5))
data.unstack() # 相当于转置 / data.stack(dropna=False) # 列变行
data['new_name'] = 'abc' # 新建列并赋值
data.iloc[0] == Series


-------------------
data1.add(data2, fill_value=0) # 对应位置相加 缺失值用0填补
data.rdiv(1) # 1/x 
data.sub(Series, axis='index')
pd.date_range('1/1/2010', periods=6, freq='D')

-------
data[:10] # 行位置索引切片

# 只能加入标签行索引 不能放位置行索引
data.loc['index_name'] # data.loc['index_name':'index_name', 'col_name':'col_name']
data.loc['index_name':'index_name']

data.iloc[6] # 只能位置行索引 data.iloc[0:6]
data.iloc[6, 6]
data.iloc[:, 1:] # .iloc[:,0] [[行]，[列]]

data.ix['index_name'] # data.ix[6] # 切片data.ix[0:6] 
data.ix['index_name', 'col_name']  # data.ix[6, 6] 既能标签 又能位置行索引
data.ix['index_name', 6]

--------------------------------------------------
# 分层索引8.1
data.index.names = ['key1', 'key2']
data.columns.names = ['col_name1', 'col_name2']
data.swaplevel('key1', 'key2') # 交换一二层索引
data.sort_index(level=1) # key1是level=0，key2是level=1

data.sum(level='key2/col_name', axis=)
data.set_index(['col_name1', 'col_name2'], drop=False) # 已有列作为多层索引
data.reset_index(drop=Ture) # 重新设置索引0-n  把原来的多层级索引变为列
# 删除原来的index
--------------------------------------------------
data.sum() # 同一列的所有行求和 / data.sum(axis=1) 同一行的所有列求和
data.mean(axis=1, skipna=True) # 跳过缺失值
data.idxmax() # 返回每列的最大值
data.describe(percentiles=[.10,.20], include=np.number)
data['col_name'].value_counts() # 该列的取值统计 
data['col_name'].mode() # 

-------------------------------
data.drop_duplicates(subset=None, keep='first', inplace=False)
df.drop_duplicats(subset = ['price','cnt'],keep='last',inplace=True)
s.index.is_unique # True/False

-----------------------
data.sort_index(axis=1, ascending=False)
data.sort_values(by=['col_name','col_name']) # s.sort_values() # 缺失值排在最后
data.rank(axis=0, ascending=True, method='first/max') # 

--------------------------------
import pickle
data = pd.read_pickle('name')
data = pd.read_csv('name')
data.to_pickle('mydata.pkl')  # pd.DateFrame


-----------------------
df = df.select_dtypes(exclude=['object']) # 除去object
df.col_name = df.col_name.astype(float)

------------------------------
groupby


------------------------------------
apply
data.apply(func, axis=1, index=['col_name',], result_type='') # axis=1 为列之间的统计，每行有一个统计量
# result_type{'expand', 'reduce', 'broadcast', None}, default None

--------------
data.dulicated()  # 是否为重复的行True/False
data = data.drop_duplicates() # para: ['col_name', ], keep='last'

-------------
'''
	缺失值处理
'''
reset_index
data.dropna(axis=1, how=) # 默认删除包含nan的所有行/how='all'只删除完全缺失的行/axis=1删除列
data[data.notnull()]
data.isnull().sum() / len(data)
data.isnull().any()		   # data.isnull().any(axis=1) 缺失行
data.isnull().any().sum()  # data.isnull().any().sum(axis=1) 统计共有几行存在缺失值
data.drop(data.columns[np.isnan(data).any()], axis=1) # 删除带有空值的列
data = data.replace(-1, np.nan)  # data.replace([-999, -1000], [np.nan, 0])

data = data.fillna(df.mean()) # data.fillna(-999999, inplace=True)
data = data.fillna({1:0, 2:-1}) # 不同列替换缺失
# fillna 参数

------------------------------------
'''
	连接
'''
data1 = data1.merge(data2, on=['col_name', 'col_name'], how='left') # on = ['col_name', ……]
data1 = data1.merge(data2, left_on='data1_col_name', right_on='data2_col_name',how='inner', suffixes=('_left', '_right'))
pd.merge(data1, data2, left_on='col_name', right_index=True) # 右表用索引进行连接 left_index/right_index
# outer join就是left join和right join同时应用的效果
# how: {'left', 'right', 'outer', 'inner'}, default 'inner'

data1.join([data2,], how='left') # 利用相同索引连接 on='key'

data = np.concatenate([arr1, arr2], axis=1) # 横向连接


pd.concat([s1, s2], axis=1) # 将列拼接在一起（行数相同）
pd.concat([s1, s2], axis=0, ignore_index=False)
pd.concat([s1, s2], axis=1, keys=['col_name1', 'col_name2']) # 竖向连接 pd.concat([s1, s2], axis=1) # 横向连接
pd.concat({'col_name1': data1, 'col_name2': data2}, axis=1)
data1.combine_first(data2) # 索引拼接先取s1 若空缺填补s2

----------------------------
'''
	分箱
'''
bins = pd.cut(Series, [10, 20, 30, 40], right=False, labels=[4-labels]) #右开
bins = pd.cut(Series, 4, precision=2) # 平均分4箱，小数点后两位
bins = pd.qcut(Series, 4) # 依据百分数等量划分
bins.codes
bins.categories
pd.value_counts(bins)

# indicator 多类别分类
dummies = data.get_dummies(data['col_name'], prefix='perfix_') # 类别变量0-1
dummies = pd.get_dummies(pd.cut(Series, []))
data[['col_name', ]].join(dummies) # 合并

indicator = pd.unique([])
dummies = pd.DataFrame(np.zeros(len(data), len(indicator)), columns=indicator)
for i, col in enumerate(data.col_name):
    indices = dummies.columns.get_indexer(gen.split('|'))
    dummies.iloc[i, indices] = 1
data = data.join(dummies.add_prefix('pre_'))


---------------------
'''
	特征过滤
'''
from sklearn.feature_selection import VarianceThreshold 
selectot = VarianceThreshold()
X_var0 = selectot.fit_transform(X) # 过滤方差为0的列
X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X) # X.var()表示方差 取方差中位数 筛选掉一半特征

'''
	特征重要性
'''
from sklearn.ensemble import RandomForestClassifier as RFC
forest = RFC(n_estimators=10, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1] # 返回的是数值从小到大的索引值
for f in range(X.shape[1]):  # feat_labels = X.columns
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

'''
	模型训练
'''
from sklearn.model_selection import cross_val_score
cross_val_score(RFC(n_estimators=10, random_state=0), X, y, cv=5).means()

--------------------------------------------------
**************************************************
--------------------------------------------------
import numpy as np
arr = np.array([])	# 一维array shape(n,) 
np.random.randn(5) # array([0-1的五个随机数])
arr = np.array(['22', '33'], dtype=np.string_)
np.arange(3) # array([0, 1, 2]) # np.arange(6).reshape((2, 3))
arr.ndim
arr.shape
arr.dtype
arr.astype(np.float64)# np.int32/np.string_/float
arr.copy()
arr.T
arr.reshape((2,5))
np.dot(arr.T, arr) # arr.T * arr
arr[:2]		# arr[1, :2], arr[:2, 1:]
arr[arr < 0] = 0 # arr[arr == 'Bob', 2:]
arr[0] # 第0维
---------------------------
np.random.rand(3, 6) # [0, 1)
np.random.randn(3, 6) # 标准正态分布 均值0 标准差1

---------------
np.sign(data) # 指示函数分正负
np.abs(data)

-------------
# Number of each type of column
df.dtypes.value_counts()
df = pd.get_dummies(df) # onehot

# Align the training and testing data, keep only columns present in both dataframes
train, test = train.align(test, join = 'inner', axis = 1)

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# KDE plot of loans that were repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')


# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
# array([20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.])
age_groups  = age_data.groupby('YEARS_BINNED').mean()