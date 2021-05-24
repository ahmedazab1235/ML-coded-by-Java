# ML-coded-by-Java

ML coded by Java 


1- Isotonic regression
Isotonic regression belongs to the family of regression algorithms. Formally isotonic regression is a problem where given a finite set of real numbers Y=y1,y2,...,yn representing observed responses and X=x1,x2,...,xn the unknown response values to be fitted finding a function that minimises

f(x)=∑i=1nwi(yi−xi)2(1)

with respect to complete order subject to x1≤x2≤...≤xn where wi are positive weights. The resulting function is called isotonic regression and it is unique. It can be viewed as least squares problem under order restriction. Essentially isotonic regression is a monotonic function best fitting the original data points.

We implement a pool adjacent violators algorithm which uses an approach to parallelizing isotonic regression. The training input is a DataFrame which contains three columns label, features and weight. Additionally IsotonicRegression algorithm has one optional parameter called isotonic defaulting to true. This argument specifies if the isotonic regression is isotonic (monotonically increasing) or antitonic (monotonically decreasing).

Training returns an IsotonicRegressionModel that can be used to predict labels for both known and unknown features. The result of isotonic regression is treated as piecewise linear function. The rules for prediction therefore are:

If the prediction input exactly matches a training feature then associated prediction is returned. In case there are multiple predictions with the same feature then one of them is returned. Which one is undefined (same as java.util.Arrays.binarySearch).
If the prediction input is lower or higher than all training features then prediction with lowest or highest feature is returned respectively. In case there are multiple predictions with the same feature then the lowest or highest is returned respectively.
If the prediction input falls between two training features then prediction is treated as piecewise linear function and interpolated value is calculated from the predictions of the two closest features. In case there are multiple values with the same feature then the same rules as in previous point are used.

2- Random forest regression
Random forests are a popular family of classification and regression methods. More information about the spark.ml implementation can be found further in the section on random forests.

Examples

The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset, and then evaluate on the held-out test set. We use a feature transformer to index categorical features, adding metadata to the DataFrame which the tree-based algorithms can recognize.
