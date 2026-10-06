
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

codecademy = pd.read_csv('codecademy.csv')

print(codecademy.head(5))

plt.scatter(codecademy.completed, codecademy.score)

plt.show()
model = sm.OLS.from_formula('score~completed', data = codecademy)
results = model.fit()
print(results.params)
plt.scatter(codecademy.completed, codecademy.score)
plt.plot(codecademy.completed, results.predict(codecademy.completed))

plt.show()
plt.clf()

newdata = {'completed':[20]}
new_df = pd.DataFrame(newdata)
score_pred = results.predict(new_df)
print(score_pred)

fitted_values = results.predict(codecademy)
print(fitted_values)

residuals = results.resid
print(residuals)
plt.hist(residuals)
plt.show()
plt.clf()

 
plt.scatter(fitted_values, residuals)
plt.show()
plt.clf()
sns.boxplot(x='lesson', y = 'score', data = codecademy)
plt.show()
#
model = sm.OLS.from_formula('score~lesson', data = codecademy)
results = model.fit()
print(results.params)

mean_A = np.mean(codecademy.score[codecademy.lesson =='Lesson A'])
mean_B = np.mean(codecademy.score[codecademy.lesson == 'Lesson B'])
print(mean_A)
print(mean_B)
print(mean_A - mean_B)
sns.lmplot(x = 'completed',y = 'score', hue = 'lesson', data = codecademy)
plt.show()
