suppressPackageStartupMessages({
  library(magrittr)
  library(tidyverse)
  library(conflicted)
})
conflicts_prefer(dplyr::filter())
options(dplyr.width = Inf)


# load and reshape data --------------------------------------------------------
train_df = bind_cols(
  read_csv('data/moons/y_train.csv'),
  read_csv('data/moons/X_train.csv'),
)

test_df = bind_cols(
  read_csv('data/moons/y_test.csv'),
  read_csv('data/moons/x_test.csv'),
)

x_train = train_df %>% select(-y)
x_test  = test_df %>% select(-y)
y_train = train_df %>% select(y)
y_test  = test_df %>% select(y)


# plot data + standard lr model ------------------------------------------------
with(train_df, plot(x1, x2, col=y + 1, pch=19, asp=1, main='LR'))

lr = glm(y ~ x1 + x2, data=train_df, family = binomial)
lr_y_hat_train = as.integer(predict(lr, x_train, type='response') > 0.5)
lr_y_hat_test = as.integer(predict(lr, x_test, type='response') > 0.5)
print(str_glue("LR train accuracy: {100 - (mean(abs(lr_y_hat_train - y_train[['y']])) * 100)} %"))
print(str_glue("LR test accuracy: {100 - (mean(abs(lr_y_hat_test - y_test[['y']])) * 100)} %"))

pred_data = with(train_df, tibble(
  x1 = seq(min(x1), max(x1), length.out=100),
  x2 = seq(min(x2), max(x2), length.out=100)
)) %>%
  tidyr::expand(x1, x2)
lr_probs = predict(lr, pred_data, type='response')
lr_preds = as.integer(lr_probs > 0.5)
with(pred_data, points(x=x1, y=x2, col=lr_preds + 1, cex=0.3, pch=15))


# activation functions ---------------------------------------------------------
sigmoid = function(x) {
  1 / (1 + exp(-x))
}

sigmoid_prime = function(x) {
  sigmoid(x) * (1 - sigmoid(x))
}

tanh_prime = function(x) {
  1 - tanh(x)^2
}

relu = function(x) {
  y = x
  i = y < 0
  y[i] = 0
  y
}

relu_prime = function(x) {
  y = x
  i = y < 0
  y[i] = 0
  y[!i] = 1
  y
}

leaky_relu = function(x) {
  # max(0.01 * x, x)
  # if (x < 0.01 * x) 0.01 * x else x
  y = x
  i = y < 0.01 * y
  y[i] = 0.01 * y[i]
  y
}

leaky_relu_prime = function(x) {
  # if (x < 0) 0.01 else 1
  y = x
  i = y < 0
  y[i] = 0.01
  y[!i] = 1
  y
}

g = function(activation_function) {
  activation_function %>%
    switch(
      'sigmoid' = sigmoid,
      'tanh' = tanh,
      'relu' = relu,
      'leaky_relu' = leaky_relu
    )
}

g_prime = function(activation_function) {
  activation_function %>%
    switch(
      'sigmoid' = sigmoid_prime,
      'tanh' = tanh_prime,
      'relu' = relu_prime,
      'leaky_relu' = leaky_relu_prime
    )
}

forward_propagation = function(X, W1, b1, W2, b2, activation_function) {
  Z1 = cbind(b1, W1) %*% rbind(1, X)
  A1 = g(activation_function)(Z1)
  Z2 = cbind(b2, W2) %*% rbind(1, A1)
  A2 = sigmoid(Z2)
  # if (n2 > 1) {
  #   A2 = apply(A2, 2, function(x) x / sum(x))
  # }
  return(list(
    'Z1' = Z1,
    'A1' = A1,
    'Z2' = Z2,
    'A2' = A2
  ))
}

back_propagation = function(X, Y, Z1, A1, W2, Z2, A2, activation_function) {
  m = nrow(X)
  dZ2 = A2 - Y
  dW2 = dZ2 %*% t(A1) / m
  db2 = sum(dZ2) / m
  dZ1 = t(W2) %*% dZ2 * g_prime(activation_function)(Z1)
  dW1 = dZ1 %*% t(X) / m
  db1 = sum(dZ1) / m
  return(list(
    'db1' = db1,
    'dW1' = dW1,
    'db2' = db2,
    'dW2' = dW2
  ))
}

make_predictions = function(model, X) {
  probs = forward_propagation(
    X = X,
    W1 = model[['W1']],
    b1 = model[['b1']],
    W2 = model[['W2']],
    b2 = model[['b2']],
    activation_function = model[['activation_function']]
  )[['A2']]
  as.integer(probs > 0.5)
}

build_model = function(x_train, y_train, x_test, y_test, n_hidden_nodes,
                       activation_function = 'sigmoid',
                       alpha= 0.05, max_iterations=100, min_improvement=0.00001,
                       print_metrics=FALSE) {

  if (0) {
    n_hidden_nodes = 5
    activation_function = 'sigmoid'
    alpha = 0.05
    max_iterations=10
    min_improvement=0.01
    print_metrics=TRUE
  }

  m = nrow(x_train)
  n0 = ncol(x_train)
  n1 = n_hidden_nodes
  n2 = ncol(y_train)

  # reshape data ---------------------------------------------------------------
  Y_train = t(as.matrix(y_train, nrow = 1))
  X_train = t(as.matrix(x_train))
  Y_test = t(as.matrix(y_test, nrow = 1))
  X_test = t(as.matrix(x_test))

  # initialize parameters ------------------------------------------------------
  W1 = matrix(rnorm(n1 * n0), nrow=n1, ncol=n0)
  b1 = matrix(0, nrow=n1, ncol=1)
  W2 = matrix(rnorm(n2 * n1), nrow=n2, ncol=n1)
  b2 = matrix(0, nrow=n2, ncol=1)

  stopifnot(dim(Y_train) == c(1, m))
  stopifnot(dim(X_train) == c(n0, m))
  stopifnot(dim(W1) == c(n1, n0))
  stopifnot(dim(b1) == c(n1, 1))
  stopifnot(dim(W2) == c(n2, n1))
  stopifnot(dim(b2) == c(n2, 1))

  cost = rep(NA, max_iterations)
  delta_cost = rep(NA, max_iterations)

  # gradient descent -----------------------------------------------------------
  for (i in 1:max_iterations) { # i=1

    # forward propagation
    fprop = forward_propagation(X_train, W1, b1, W2, b2, activation_function)
    Z1 = fprop[['Z1']]
    A1 = fprop[['A1']]
    Z2 = fprop[['Z2']]
    A2 = fprop[['A2']]

    stopifnot(dim(Z1) == c(n1, m))
    stopifnot(dim(A1) == c(n1, m))
    stopifnot(dim(Z2) == c(n2, m))
    stopifnot(dim(A2) == dim(Z2))

    cost[i] = -sum(Y_train * log(A2) + (1 - Y_train) * log(1 - A2)) / m

    # back propagation
    bprop = back_propagation(X_train, Y_train, Z1, A1, W2, Z2, A2, activation_function)
    dW1 = bprop[['dW1']]
    db1 = bprop[['db1']]
    dW2 = bprop[['dW2']]
    db2 = bprop[['db2']]

    stopifnot(dim(dW1) == dim(W1))
    stopifnot(dim(db1) == dim(b1))
    stopifnot(dim(dW2) == dim(W2))
    stopifnot(dim(db2) == dim(b2))

    # # Add regularization terms (b1 and b2 don't have regularization terms)
    # dW2 += reg_lambda * W2
    # dW1 += reg_lambda * W1

    # gradient descent parameter update
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    cost[i] = -sum(Y_train * log(A2) + (1 - Y_train) * log(1 - A2)) / m

    delta_cost[i] = if (i > 1) cost[i] - cost[i-1] else NA

    if (print_metrics & (i==1 | i %% 100 == 0)) {
      print(str_glue("Cost after iteration {i}: {round(cost[i], 5)} (delta: {round(delta_cost[i], 5)})"))
    }

    if (i == max_iterations) {
      print(str_glue('stopping at iteration {i}: max iterations ({max_iterations}) reached'))
      break
    }

    if (i > 1000) {
      if (delta_cost[i] > -min_improvement) {
        print(str_glue('stopping at iteration {i}: delta cost ({delta_cost[i]}) > -min_improvement (-{min_improvement})'))
        break
      }
    }

  }

  model = list(
    'W1' = W1,
    'b1' = b1,
    'W2' = W2,
    'b2' = b2,
    cost = cost[1:i],
    delta_cost = delta_cost[1:i],
    max_iterations=max_iterations,
    activation_function=activation_function,
    alpha=alpha,
    n_hidden_nodes=n_hidden_nodes
  )

  # predictions ----------------------------------------------------------------
  Y_hat_train = make_predictions(model, X_train)
  Y_hat_test  = make_predictions(model, X_test)

  # print metrics --------------------------------------------------------------
  if (print_metrics) {
    print(str_glue("NN train accuracy: {100 - (mean(abs(Y_hat_train - Y_train)) * 100)} %"))
    print(str_glue("NN test accuracy: {100 - (mean(abs(Y_hat_test - Y_test)) * 100)} %"))
  }

  return(model)

}

# debugonce(tanh)
# debugonce(tanh_prime)
# debugonce(build_model)

set.seed(42)
nn = build_model(
  x_train=x_train,
  y_train=y_train,
  x_test=x_test,
  y_test=y_test,
  n_hidden_nodes=10,
  # activation_function='sigmoid',
  # activation_function='tanh',
  activation_function='relu',
  # activation_function='leaky_relu',
  alpha=0.001,
  max_iterations=10000,
  print_metrics=TRUE
)

with(nn, plot(seq_along(cost), cost, type='l', main='cost'))

nn_preds = make_predictions(nn, t(as.matrix(pred_data)))
title = with(nn, str_glue('NN: {max_iterations} iters, {activation_function}, alpha {alpha}, {n_hidden_nodes} hidden nodes'))
with(train_df, plot(x1, x2, col=y + 1, pch=19, asp=1, main=title))
with(pred_data, points(x=x1, y=x2, col=(as.numeric(nn_preds) > 0.5) + 1, cex=0.3, pch=15))

print(head(cbind(cost=nn[['cost']], delta=nn[['delta_cost']])))
print(tail(cbind(cost=nn[['cost']], delta=nn[['delta_cost']])))


