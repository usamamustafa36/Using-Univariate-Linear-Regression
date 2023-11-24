import pandas as pd  # Import the pandas library for data manipulation

# Load the dataset from the specified file path
file_path = r"D:\ML\Medal_Prediction\teams.csv"
teams = pd.read_csv(file_path)

teams  # Display the loaded dataset

# Select specific columns for analysis: team, country, year, athletes, prev_medals, medals
teams = teams[["team", "country", "year", "athletes", "prev_medals", "medals"]]

teams  # Display the dataset with selected columns

# Identify numeric columns and calculate their correlation with the 'medals' column
numeric_columns = teams.select_dtypes(include=['float', 'int']).columns
correlation_medals = teams[numeric_columns].corr()['medals']
print(correlation_medals)  # Display the correlation coefficients

import seaborn as sns  # Import the seaborn library for visualization

# Plot a scatterplot to visualize the relationship between 'prev_medals' and 'medals'
sns.lmplot(x='prev_medals', y='medals', data=teams, fit_reg=True, ci=None)

# Plot a histogram to display the distribution of the 'medals' column
teams.plot.hist(y="medals")

# Display rows with any missing values in the dataset
teams[teams.isnull().any(axis=1)].head(20)

# Remove rows with missing values from the dataset
teams = teams.dropna()

teams  # Display the cleaned dataset

teams.shape  # Display the shape (rows, columns) of the cleaned dataset

# Create training and testing datasets based on the 'year' criterion
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

train.shape  # Display the shape of the training dataset (about 80% of the data)
test.shape  # Display the shape of the testing dataset (about 20% of the data)

from sklearn.linear_model import LinearRegression  # Import Linear Regression model from scikit-learn

reg = LinearRegression()  # Initialize Linear Regression model

predictor = ["prev_medals"]  # Set the predictor variable

# Train the Linear Regression model using the training dataset
reg.fit(train[predictor], train["medals"])

target = "medals"  # Define the target variable

predictions = reg.predict(test[predictor])  # Make predictions using the test dataset

predictions  # Display the predicted values

test["predictions"] = predictions  # Add predictions to the test dataset

test  # Display the test dataset with predictions

test.loc[test["predictions"] < 0, "predictions"] = 0  # Handle negative predictions

test["predictions"] = test["predictions"].round()  # Round predictions to integers

test  # Display the test dataset with rounded predictions

from sklearn.metrics import mean_absolute_error  # Import Mean Absolute Error metric

# Calculate the Mean Absolute Error between actual and predicted values
error = mean_absolute_error(test["medals"], test["predictions"])

teams.describe()["medals"]  # Display descriptive statistics for the 'medals' column

test[test["team"] == "PAK"]  # Display data for a specific team ('PAK')

test[test["team"] == "IND"]  # Display data for another specific team ('IND')

# ...Repeat for other specific teams (e.g., 'AFG', 'USA')...

errors = (test["medals"] - test["predictions"]).abs()  # Calculate absolute errors

errors  # Display the absolute errors

error_by_team = errors.groupby(test["team"]).mean()  # Calculate mean error by team

error_by_team  # Display mean error by team

medals_by_team = test["medals"].groupby(test["team"]).mean()  # Calculate mean medals by team

error_ratio = error_by_team / medals_by_team  # Calculate error ratio by team

error_ratio  # Display error ratio by team

error_ratio = error_ratio[np.isfinite(error_ratio)]  # Remove infinite values from error ratio

error_ratio.plot.hist()  # Plot a histogram of error ratio distribution

error_ratio.sort_values()  # Sort and display error ratio values
