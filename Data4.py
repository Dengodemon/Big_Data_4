#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np

street = np.array([80, 98, 75, 91, 78])
garage = np.array([100, 82, 105, 89, 102])
correlate = np.corrcoef(street, garage)[0, 1] # Вычисление коэффициента корреляции Пирсона
print(f"Коэффициент корреляции Пирсона: {correlate}")


# In[3]:


import matplotlib.pyplot as plt

street = [80, 98, 75, 91, 78]
garage = [100, 82, 105, 89, 102]
plt.scatter(street, garage, c='b', label='Диаграмма рассеяния')
plt.title('Диаграмма рассеяния между Улицей и Гаражом') 
plt.xlabel('Улица')
plt.ylabel('Гараж') 
plt.legend()
plt.grid(True)
plt.show()


# In[7]:


import pandas as pd 
import numpy as np

data = pd.read_csv("school-shootings-data.csv")
data = data.fillna(method='ffill')# обработка данных
print(data.describe())


# In[2]:


import pandas as pd 
import numpy as np

data = pd.read_csv("school-shootings-data.csv") 
pd.set_option('display.max_columns', None) 
target = 'killed' # Выбор переменной  
n_data = data.select_dtypes(include=[np.number])# Только числа
matrix = n_data.corr() # Построение корреляционной матриц
print(cmatrix)
correlate = matrix[target].sort_values(ascending=False)[1:].idxmax()# Нахождение наиболее коррелирующей переменной 
print(f"\nНаиболее коррелирующая переменная с '{target}' это '{correlate}'")


# In[17]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
data = pd.read_csv("school-shootings-data.csv")
target_variable = 'killed' 
feature_variable = 'casualties'
data.dropna(inplace=True)
# Реализация регрессии
X = data[feature_variable].values 
y = data[target_variable].values

# Вычисление наклона и сдвига
slope = np.polyfit(X, y, 1)
intercept = np.polyfit(X, y, 1)

# Вычисление прогнозных значений
y_pred = slope * X + intercept

# Вычисление среднеквадратичной ошибки
mse = ((y - y_pred) ** 2).mean()

print("Наклон (коэффициент наклона):", slope) 
print("Сдвиг (свободный член):", intercept) 
print("Среднеквадратичная ошибка:", mse)
plt.scatter(X, y, label='Наблюдения', color='b') 
plt.plot(X, y_pred, label='Регрессия', color='r') 
plt.title('Регрессия') 
plt.xlabel(feature_variable) 
plt.ylabel(target_variable)
plt.legend() 


# In[35]:


import pandas as pd
from scipy.stats import f_oneway 
anova = []
data = pd.read_csv("insurance.csv")
regions = data['region'].unique()# Уникальные регионы 
print("Уникальные регионы:", regions)
for region in regions:# ANOVA тест для регионов
    subset = data[data['region'] == region] ['bmi'] 
    anova.append((region, subset))
f_statistic, p_value = f_oneway(*[subset for region, subset in anova])
print("Результаты ANOVA теста:") 
print(f"F-статистика: {f_statistic:.4f}") 
print(f"P-значение: {p_value:.4f}")


# In[36]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

data = pd.read_csv("insurance.csv")
model = ols('bmi ~ region', data=data).fit() # Однофакторный ANOVA тест
table = sm.stats.anova_lm(model, typ=2)
print("Результаты ANOVA теста:") 
print(table)


# In[43]:


import pandas as pd
from itertools import combinations 
from scipy.stats import ttest_ind

data = pd.read_csv("insurance.csv")
regions = data['region'].unique()
alpha = 0.05
bonfer = alpha / len(regions)
region = list(combinations(regions, 2)) 
differences = []
for region1, region2 in region:
    subset1 = data[data['region'] == region1]['bmi'] 
    subset2 = data[data['region'] == region2]['bmi']
    t_statistic, p_value = ttest_ind(subset1, subset2)# t-критерий Стьюдента
    if p_value < bonfer: # t-критерий Стьюдента
        differences.append((region1, region2, p_value))
print("Рразличия между парами регионов:")
for region1, region2, p_value in differences: 
    print(f"{region1} и {region2}: p-значение = {p_value:.4f}")


# In[45]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison 
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv")
mc = MultiComparison(data['bmi'], data['region']) # Значимые различия между парами регионов
result = mc.tukeyhsd()
print(result)
fig = result.plot_simultaneous() 
plt.show()


# In[27]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols 
data = pd.read_csv("insurance.csv")

formula = 'bmi ~ region + sex + region:sex'# Двухфакторный ANOVA тест
model = sm.formula.ols(formula, data=data).fit() 
table = sm.stats.anova_lm(model, typ=2)
print(table)


# In[29]:


import pandas as pd
from statsmodels.stats.multicomp import MultiComparison 
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv") 
data['combination'] = data['region'] + ' / ' + data['sex'] # Комбинированный столбцец
mc = MultiComparison(data['bmi'], data['combination']) 
result = mc.tukeyhsd()
print(result)
result.plot_simultaneous() 
plt.show()


# In[ ]:




