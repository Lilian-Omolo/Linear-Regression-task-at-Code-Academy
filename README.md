# Linear-Regression-task-at-Code-Academy

# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Read in the data
codecademy = pd.read_csv('codecademy.csv')

# Print the first five rows
print(codecademy.head(5))

# Create a scatter plot of score vs completed
plt.scatter(codecademy.completed, codecademy.score)

# Show then clear plot
plt.show()
# Fit a linear regression to predict score based on prior lessons completed
model = sm.OLS.from_formula('score~completed', data = codecademy)
results = model.fit()
print(results.params)

# Intercept interpretation:expected value of the outcome variable when the predictor variable is zero

# Slope interpretation: expected difference in the outcome variable for one unit increase in predictor variable

# Plot the scatter plot with the line on top
plt.scatter(codecademy.completed, codecademy.score)
plt.plot(codecademy.completed, results.predict(codecademy.completed))

# Show then clear plot
plt.show()
plt.clf()
# Predict score for learner who has completed 20 prior lessons
#data in dict convert to dataframe then use the prediction
newdata = {'completed':[20]}
new_df = pd.DataFrame(newdata)
score_pred = results.predict(new_df)
print(score_pred)

# Calculate fitted values
# add the constant 
fitted_values = results.predict(codecademy)
print(fitted_values)

# Calculate residuals
residuals = results.resid
print(residuals)
# Check normality assumption
plt.hist(residuals)
# Show then clear the plot
plt.show()
plt.clf()

# Check homoscedasticity assumption
# no pattern 
plt.scatter(fitted_values, residuals)
plt.show()
# Show then clear the plot
plt.clf()
# Create a boxplot of score vs lesson
sns.boxplot(x='lesson', y = 'score', data = codecademy)
# Show then clear plot
plt.show()
# Fit a linear regression to predict score based on which lesson they took
# creating the score vs lesson model
model = sm.OLS.from_formula('score~lesson', data = codecademy)
results = model.fit()
print(results.params)

# Calculate and print the group means and mean difference (for comparison)
# mean_score_lessonA = np.mean(codecademy.score[codecademy.lesson == 'Lesson A'])
mean_A = np.mean(codecademy.score[codecademy.lesson =='Lesson A'])
mean_B = np.mean(codecademy.score[codecademy.lesson == 'Lesson B'])
print(mean_A)
print(mean_B)
print(mean_A - mean_B)
# Use `sns.lmplot()` to plot `score` vs. `completed` colored by `lesson`
sns.lmplot(x = 'completed',y = 'score', hue = 'lesson', data = codecademy)
plt.show()
