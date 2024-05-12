This section provides a brief overview of the Naive Bayes algorithm and the Iris flowers dataset that we will use in this tutorial.

Naive Bayes
Bayes’ Theorem provides a way that we can calculate the probability of a piece of data belonging to a given class, given our prior knowledge. Bayes’ Theorem is stated as:

P(class|data) = (P(data|class) * P(class)) / P(data)
Where P(class|data) is the probability of class given the provided data.

For an in-depth introduction to Bayes Theorem, see the tutorial:

A Gentle Introduction to Bayes Theorem for Machine Learning
Naive Bayes is a classification algorithm for binary (two-class) and multiclass classification problems. It is called Naive Bayes or idiot Bayes because the calculations of the probabilities for each class are simplified to make their calculations tractable.

Rather than attempting to calculate the probabilities of each attribute value, they are assumed to be conditionally independent given the class value.

This is a very strong assumption that is most unlikely in real data, i.e. that the attributes do not interact. Nevertheless, the approach performs surprisingly well on data where this assumption does not hold.


Iris Flower Species Dataset
In this tutorial we will use the Iris Flower Species Dataset.

The Iris Flower Dataset involves predicting the flower species given measurements of iris flowers.

It is a multiclass classification problem. The number of observations for each class is balanced. There are 150 observations with 4 input variables and 1 output variable. The variable names are as follows:

Sepal length in cm.
Sepal width in cm.
Petal length in cm.
Petal width in cm.
Class
A sample of the first 5 rows is listed below





First we will develop each piece of the algorithm in this section, then we will tie all of the elements together into a working implementation applied to a real dataset in the next section.

This Naive Bayes tutorial is broken down into 5 parts:

Step 1: Separate By Class.
Step 2: Summarize Dataset.
Step 3: Summarize Data By Class.
Step 4: Gaussian Probability Density Function.
Step 5: Class Probabilities.
These steps will provide the foundation that you need to implement Naive Bayes from scratch and apply it to your own predictive modeling problems.


Step 1: Separate By Class
We will need to calculate the probability of data by the class they belong to, the so-called base rate.

This means that we will first need to separate our training data by class. A relatively straightforward operation.

We can create a dictionary object where each key is the class value and then add a list of all the records as the value in the dictionary.

Below is a function named separate_by_class() that implements this approach. It assumes that the last column in each row is the class value.


----
Putting this all together, we can test our separate_by_class() function on the contrived dataset.



-----
Running the example sorts observations in the dataset by their class value, then prints the class value followed by all identified records.




Step 2: Summarize Dataset
We need two statistics from a given set of data.

We’ll see how these statistics are used in the calculation of probabilities in a few steps. The two statistics we require from a given dataset are the mean and the standard deviation (average deviation from the mean).

The mean is the average value and can be calculated as:

mean = sum(x)/n * count(x)
Where x is the list of values or a column we are looking.

Below is a small function named mean() that calculates the mean of a list of numbers.

# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))
The sample standard deviation is calculated as the mean difference from the mean value. This can be calculated as:

standard deviation = sqrt((sum i to N (x_i – mean(x))^2) / N-1)
You can see that we square the difference between the mean and a given value, calculate the average squared difference from the mean, then take the square root to return the units back to their original value.

Below is a small function named standard_deviation() that calculates the standard deviation of a list of numbers. You will notice that it calculates the mean. It might be more efficient to calculate the mean of a list of numbers once and pass it to the standard_deviation() function as a parameter. You can explore this optimization if you’re interested later.

from math import sqrt
 
# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
We require the mean and standard deviation statistics to be calculated for each input attribute or each column of our data.

We can do that by gathering all of the values for each column into a list and calculating the mean and standard deviation on that list. Once calculated, we can gather the statistics together into a list or tuple of statistics. Then, repeat this operation for each column in the dataset and return a list of tuples of statistics.

Below is a function named summarize_dataset() that implements this approach. It uses some Python tricks to cut down on the number of lines required.

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries
The first trick is the use of the zip() function that will aggregate elements from each provided argument. We pass in the dataset to the zip() function with the * operator that separates the dataset (that is a list of lists) into separate lists for each row. The zip() function then iterates over each element of each row and returns a column from the dataset as a list of numbers. A clever little trick.

We then calculate the mean, standard deviation and count of rows in each column. A tuple is created from these 3 numbers and a list of these tuples is stored. We then remove the statistics for the class variable as we will not need these statistics.

Let’s test all of these functions on our contrived dataset from above. Below is the complete example.

Running the example prints out the list of tuples of statistics on each of the two input variables.

Interpreting the results, we can see that the mean value of X1 is 5.178333386499999 and the standard deviation of X1 is 2.7665845055177263.


Now we are ready to use these functions on each group of rows in our dataset.




Step 3: Summarize Data By Class
We require statistics from our training dataset organized by class.

Above, we have developed the separate_by_class() function to separate a dataset into rows by class. And we have developed summarize_dataset() function to calculate summary statistics for each column.

We can put all of this together and summarize the columns in the dataset organized by class values.

Below is a function named summarize_by_class() that implements this operation. The dataset is first split by class, then statistics are calculated on each subset. The results in the form of a list of tuples of statistics are then stored in a dictionary by their class value.



Running this example calculates the statistics for each input variable and prints them organized by class value. Interpreting the results, we can see that the X1 values for rows for class 0 have a mean value of 2.7420144012.




Step 4: Gaussian Probability Density Function
Calculating the probability or likelihood of observing a given real-value like X1 is difficult.

One way we can do this is to assume that X1 values are drawn from a distribution, such as a bell curve or Gaussian distribution.

A Gaussian distribution can be summarized using only two numbers: the mean and the standard deviation. Therefore, with a little math, we can estimate the probability of a given value. This piece of math is called a Gaussian Probability Distribution Function (or Gaussian PDF) and can be calculated as:

f(x) = (1 / sqrt(2 * PI) * sigma) * exp(-((x-mean)^2 / (2 * sigma^2)))
Where sigma is the standard deviation for x, mean is the mean for x and PI is the value of pi.

Below is a function that implements this. I tried to split it up to make it more readable.


Running it prints the probability of some input values. You can see that when the value is 1 and the mean and standard deviation is 1 our input is the most likely (top of the bell curve) and has the probability of 0.39.

We can see that when we keep the statistics the same and change the x value to 1 standard deviation either side of the mean value (2 and 0 or the same distance either side of the bell curve) the probabilities of those input values are the same at 0.24.



Now that we have all the pieces in place, let’s see how we can calculate the probabilities we need for the Naive Bayes classifier.




Step 5: Class Probabilities
Now it is time to use the statistics calculated from our training data to calculate probabilities for new data.

Probabilities are calculated separately for each class. This means that we first calculate the probability that a new piece of data belongs to the first class, then calculate probabilities that it belongs to the second class, and so on for all the classes.

The probability that a piece of data belongs to a class is calculated as follows:

P(class|data) = P(X|class) * P(class)
You may note that this is different from the Bayes Theorem described above.

The division has been removed to simplify the calculation.

This means that the result is no longer strictly a probability of the data belonging to a class. The value is still maximized, meaning that the calculation for the class that results in the largest value is taken as the prediction. This is a common implementation simplification as we are often more interested in the class prediction rather than the probability.

The input variables are treated separately, giving the technique it’s name “naive“. For the above example where we have 2 input variables, the calculation of the probability that a row belongs to the first class 0 can be calculated as:

P(class=0|X1,X2) = P(X1|class=0) * P(X2|class=0) * P(class=0)
Now you can see why we need to separate the data by class value. The Gaussian Probability Density function in the previous step is how we calculate the probability of a real value like X1 and the statistics we prepared are used in this calculation.

Below is a function named calculate_class_probabilities() that ties all of this together.

It takes a set of prepared summaries and a new row as input arguments.

First the total number of training records is calculated from the counts stored in the summary statistics. This is used in the calculation of the probability of a given class or P(class) as the ratio of rows with a given class of all rows in the training data.

Next, probabilities are calculated for each input value in the row using the Gaussian probability density function and the statistics for that column and of that class. Probabilities are multiplied together as they accumulated.

This process is repeated for each class in the dataset.

Finally a dictionary of probabilities is returned with one entry for each class.


Let’s tie this together with an example on the contrived dataset.

The example below first calculates the summary statistics by class for the training dataset, then uses these statistics to calculate the probability of the first record belonging to each class.




The example below first calculates the summary statistics by class for the training dataset, then uses these statistics to calculate the probability of the first record belonging to each class.

Running the example prints the probabilities calculated for each class.

We can see that the probability of the first row belonging to the 0 class (0.0503) is higher than the probability of it belonging to the 1 class (0.0001). We would therefore correctly conclude that it belongs to the 0 class.



This section applies the Naive Bayes algorithm to the Iris flowers dataset.

The first step is to load the dataset and convert the loaded data to numbers that we can use with the mean and standard deviation calculations. For this we will use the helper function load_csv() to load the file, str_column_to_float() to convert string numbers to floats and str_column_to_int() to convert the class column to integer values.

We will evaluate the algorithm using k-fold cross-validation with 5 folds. This means that 150/5=30 records will be in each fold. We will use the helper functions evaluate_algorithm() to evaluate the algorithm with cross-validation and accuracy_metric() to calculate the accuracy of predictions.

A new function named predict() was developed to manage the calculation of the probabilities of a new row belonging to each class and selecting the class with the largest probability value.

Another new function named naive_bayes() was developed to manage the application of the Naive Bayes algorithm, first learning the statistics from a training dataset and using them to make predictions for a test dataset.

If you would like more help with the data loading functions used below, see the tutorial:

How to Load Machine Learning Data From Scratch In Python
If you would like more help with the way the model is evaluated using cross validation...


Running the example prints the mean classification accuracy scores on each cross-validation fold as well as the mean accuracy score.

We can see that the mean accuracy of about 95% is dramatically better than the baseline accuracy of 33%.


We can fit the model on the entire dataset and then use the model to make predictions for new observations (rows of data).

For example, the model is just a set of probabilities calculated via the summarize_by_class() function.


Once calculated, we can use them in a call to the predict() function with a row representing our new observation to predict the class label.



We also might like to know the class label (string) for a prediction. We can update the str_column_to_int() function to print the mapping of string class names to integers so we can interpret the prediction by the model.

Tying this together, a complete example of fitting the Naive Bayes model on the entire dataset and making a single prediction for a new observation is listed below.

Running the data first summarizes the mapping of class labels to integers and then fits the model on the entire dataset.

Then a new observation is defined (in this case I took a row from the dataset), and a predicted label is calculated. In this case our observation is predicted as belonging to class 1 which we know is “Iris-versicolor“.







