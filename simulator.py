# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
from projects.demand_simulator.project_constants import PROJECT_NAME
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

RAW_DIRECTORY = f"./projects/{PROJECT_NAME}/data/raw"
STANDARD_DIRECTORY = f"./projects/{PROJECT_NAME}/data/standard"

# %%
# Functions


def random_coefficients(n) -> np.array:
    """Generate the random coefficients for the linear part of the demand function"""

    return np.random.rand(n)


def sigmoid(price, alpha, beta):
    """_summary_

    Parameters
    ----------
    price : _type_
        x values
    alpha : _type_
        _description_
    beta : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # z = np.exp(beta * price)
    # return # alpha / (1 + z)
    return alpha * sc.special.expit(-beta * price)


def graph_sigmoid(price, sigmoid: np.array):

    plt.plot(price, sigmoid)
    return plt.show()


def demand(
    coefficients: np.array,
    feat_batch: np.array,
    alpha: float = 2,
    beta: float = 1,
    price: np.array = None,
) -> float:
    """Demand function. Computes the demand given the values of the features and price"

    Parameters
    ----------
    coefficients : 1-D np.array
        These are the fixed random coefficients of the linear part of the demand function
    feat_batch : np.array
        These are the values of the features, not including the price.
    alpha: float
        Parameter to scale the logit on the y-axis
    beta: float
        Parameter to scale the logit on the x-axis
    Returns
    -------
    float
        Evaluation of the demand function on given values of the features and price
    """

    sigmoid_values = sigmoid(price, alpha, beta)
    q = np.array(np.dot(feat_batch, coefficients) + sigmoid_values)
    # You may use np.frompyfunc(f,1,1)
    # Consider only using np.array instead than floats and arrays
    # if isinstance(q, float):
    #     return q  # round(q)
    # else:
    #    q[q < 0] = 0
    q[q < 0]
    return q  # .astype(int)


# %%

start_time = time.time()

# define model
# model = XGBRegressor()

# Simulator
number_of_features = int(
    input("With how many features, besides price, would you like to simulate?: ")
)

features = []
features.append("price")
for i in range(number_of_features):
    features.append(f"feat{i+1}")

features.append("demand")

print(features)

# val_features = features.pop()
# setting first parameters
alpha = 200
print(f"alpha: {alpha}")
beta = 0.15
print(f"beta: {beta}")
n = len(features) - 2

errors = []
y_true = []
y_pred = []
t = 0
k = 100

mu = 10  # mean of prices
standard_deviation = 5  # standard deviation of prices
batch_size = 10  # number of initial rows in training set
val_batch_size = 1024  # number of rows in validation set
low = 0.0  # lower bound of random features
high = 1.0  # upper bound of random features


# Choose the n coefficients k_i randomly in the interval [0,1]
print(f"The {n} coefficients are chosen randomly in the interval [0,1]")
coefficients = random_coefficients(n)
print(f"Coefficients: {coefficients}")

# graph of the sigmoid function to work with
interval = np.arange(start=0, step=0.2, stop=20)
sigmoid_values = sigmoid(price=interval, alpha=alpha, beta=beta)
print(f"Graph of the sigmoid function to work with. Here alpha = {alpha} and beta = {beta}.")
graph_sigmoid(interval, sigmoid_values)

# Create the training set
# Create numpy array with random uniform features between 0 and 1, prices are taken from
# a normal distribution located at mu loc=mu and standard deviation scale=standard_deviation

# TODO: Bucketize the prices

feat_batch = np.random.uniform(low=low, high=high, size=(batch_size, n))
price_batch = np.abs(np.random.normal(loc=mu, scale=standard_deviation, size=batch_size))

price_demand = sigmoid(price_batch, alpha, beta)
print("Graph of the logit-price part in demand function")
plt.scatter(price_batch, price_demand)
plt.show()


demand_column = demand(
    coefficients=coefficients, feat_batch=feat_batch, price=price_batch, alpha=alpha, beta=beta
)
random_array = np.c_[price_batch, feat_batch, demand_column]

df_training = pd.DataFrame(random_array, columns=features)

# Create a validation set
# Create numpy array with random uniform features between 0 and 1, prices are taken from
# a normal distribution located at mu loc=10 and standard deviation scale=5

val_feat_batch = np.random.uniform(low=low, high=high, size=(val_batch_size, n))
val_price_batch = np.random.normal(loc=mu, scale=standard_deviation, size=val_batch_size)

val_array = np.c_[val_price_batch, val_feat_batch]

# Compute q(val_array). Note that val_array (val_feat_batch, val_price_batch) and q (coefficients,
# alpha, beta) are fixed throughout the simulation.
q_true = demand(
    coefficients=coefficients,
    feat_batch=val_feat_batch,
    price=val_price_batch,
    alpha=alpha,
    beta=beta,
)
val_mae_errors = []
val_me_errors = []
# Simulator loop

while t < k:
    print(f"Run {t+1} in simulator loop")
    data = df_training.values
    X, y = data[:, :-1], data[:, -1]
    print(f"lenght of training set in run {t} is: {X.shape[0]}")
    # # define model
    model = XGBRegressor()
    # fit model
    model.fit(X, y)
    # Model predictions on validation set
    q_pred = model.predict(val_array)
    # performance metrics
    # Compare the demand function q with the trained model q_pred

    print(
        f"""Performance metrics \n val_mae_error: {mean_absolute_error(q_true, q_pred)} \n 
          val_me_error: {np.mean(q_true - q_pred)}"""
    )
    val_mae_errors.append(mean_absolute_error(q_true, q_pred))
    val_me_errors.append(np.mean(q_true - q_pred))

    # define new data
    new_features_values = np.random.uniform(low=low, high=high, size=n)
    new_price = np.abs(np.random.normal(loc=mu, scale=standard_deviation, size=1))

    # The new row needs to be 2D
    # TODO: consider adding batches of rows
    new_row = np.array([np.r_[new_price, new_features_values]])
    print(f"New row: {new_row}")

    # make a prediction inputing an array of new price and new features

    y_hat = model.predict(new_row)
    yhat = y_hat[0]  # round(y_hat[0])
    y_pred.append(yhat)
    # summarize prediction
    print("Predicted demand by the xgboost model: %.3f" % yhat)

    # Evaluate the demand function q in the new row
    y = demand(
        coefficients=coefficients,
        feat_batch=new_row[0][1:],
        alpha=alpha,
        beta=beta,
        price=new_row[0][0],
    )
    print(f"Value of demand function q in the new row:{y}")
    y_true.append(y)
    # Compute the error
    error = np.abs(y - yhat)
    print(f"Error: {error}")
    errors.append(error)
    new_row_training = np.r_[new_row[0], y]
    random_array = np.r_[random_array, [new_row_training]]
    df_training = pd.DataFrame(random_array, columns=features)
    t += 1

# Performance metrics
print("Performance metrics of the demand curve simulator")
print(f"starting training set size: {batch_size},")
print(f"validation batch size: {val_batch_size},")
print(f"number of simulation runs k: {k} \n")

print("MAE evolution")
plt.scatter(np.arange(k), val_mae_errors)
plt.show()

print("ME evolution")
plt.scatter(np.arange(k), val_me_errors)
plt.show()

print(f"max_demand_value: {q_true.max()}")
print(f"min_demand_value: {q_true.min()}")
print(f"mean_demand_value: {q_true.mean()}")
print(f"standard_deviation_value: {np.std(q_true)}")

print("--- %s seconds ---" % (time.time() - start_time))

# %%


"""fijar val_set: se lo pasamos en cada iteracion, plotear los errores. Agregar ME.  
Agregar condicion de parada
Probar con el precio optimo que sale de: rango de precios, calcular.

Usar expit de scipy en q scipy.special.expit(x)
en función demand() todo a arrays o usar np.frompyfunc(f,1,1)
comparar error precios optimos del modelo y de la q
comparar errores de los profits con estos precios optimos
Anadir constraint de monotonicity"""
# %%
