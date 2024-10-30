# Import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from Fig_morphology_energy_data_Sub111 import energy_area_data
import matplotlib.pyplot as plt

# Load data
area1, energy1 = energy_area_data("morphology_energy_none_zero_Sub100_Fig2.xlsx")
area1 = np.array(area1)[1:]
energy1 = np.array(energy1)[1:]

# Data preprocessing: standardization
scaler = StandardScaler()
area1 = scaler.fit_transform(area1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(area1, energy1, test_size=0.2, random_state=42)

# Create neural network model
model = Sequential()
model.add(Input(shape=(3,)))  # Use Input layer
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='linear'))

# Compile the model, choose loss function and optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# Define early stopping and model checkpoint callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)  # Change filename
]

# Ensure y_train is of numeric type
y_train = np.array(y_train).astype(np.float64)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, validation_split=0.2, callbacks=callbacks)

# Evaluate the model with the test set
y_pred = model.predict(X_test)

# Ensure y_test and y_pred are of the correct data type
y_test = np.array(y_test).astype(np.float64)
y_pred = np.array(y_pred).astype(np.float64)

# Calculate evaluation metrics
mse = mean_squared_error(y_test[:,0], y_pred[:,0])
rmse = np.sqrt(mse)
r2 = r2_score(y_test[:,0], y_pred[:,0])

print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

print(len(y_train[:,0]), len(model.predict(X_train)[:,0]), len(model.predict(X_test)[:,0]), len(y_pred[:,0]))


# Example of predicting new data (using standardized input)
new_data = np.array([[0.03,  0.15, 0.82], [0.18,  0.17, 0.65], [0.35,  0.12, 0.53], [0.39,  0.07, 0.54], [0.499,  0.499, 0.002]])
new_data_scaled = scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
print("Predictions:")
print(predictions)


# Evaluate the model with the training set
y_pred_train = model.predict(X_train)

# Ensure y_train and y_pred_train are of the correct data type
y_train = np.array(y_train).astype(np.float64)
y_pred_train = np.array(y_pred_train).astype(np.float64)

fig  = plt.figure(figsize=(4.0, 3.5)) # figsize=(4.0, 3.5)
ax = fig.add_subplot(1, 1, 1)
plt.scatter(y_train[:, 0], y_pred_train[:, 0], s = 5.5, marker='o', color = (1.0, 0.0, 0.0), label = 'Training set') 
plt.plot([-1.2, 1.2], [-1.2, 1.2], 'k--', linewidth = 0.8)   
# plt.scatter(y_test[:,0], model.predict(X_test)[:,0], s = 1.5, color = (1.0, 0.0, 0.33), label = 'Test set')
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_xticks(np.linspace(-1.2, 1.2, 7))
ax.set_yticks(np.linspace(-1.2, 1.2, 7))
ax.set_ylabel(r"Predicted value, $\gamma^{*}/\gamma_{100}$")	    
ax.set_xlabel(r"Actual value, $\gamma^{*}/\gamma_{100}$") 
plt.subplots_adjust(left = 0.2, bottom = 0.15)
legend = plt.legend(loc = 'upper left')
plt.savefig("Sub100_gamma{Sub}_gamma{100}_Train.png", dpi=600)

mae_train = mean_absolute_error(y_train[:,0], y_pred_train[:,0])
mse_train = mean_squared_error(y_train[:,0], y_pred_train[:,0])
r2_train = r2_score(y_train[:,0], y_pred_train[:,0])
rmse_train = np.sqrt(mse_train)
print(r"--------------$\gamma^{*}/\gamma_{100}$--------------")
print('Train MAE:', mae_train)
print('Train MSE:', mse_train)
print('Train R^2:', r2_train)
print('Train RMSE:', rmse_train)

plt.show() 

################################################################################

# Evaluate the model with the test set
y_pred_test = model.predict(X_test)

# Ensure y_test and y_pred_test are of the correct data type
y_test = np.array(y_test).astype(np.float64)
y_pred_test = np.array(y_pred_test).astype(np.float64)

fig  = plt.figure(figsize=(4.0, 3.5)) # figsize=(4.0, 3.5)
ax = fig.add_subplot(1, 1, 1)
plt.scatter(y_test[:,0], y_pred_test[:,0], s = 5.5, marker='^', color = (1.0, 0.0, 0.0), label = 'Test set') 
plt.plot([-1.2, 1.2], [-1.2, 1.2], 'k--', linewidth = 0.8)   
# plt.scatter(y_test[:,0], model.predict(X_test)[:,0], s = 1.5, color = (1.0, 0.0, 0.33), label = 'Test set')
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_xticks(np.linspace(-1.2, 1.2, 7))
ax.set_yticks(np.linspace(-1.2, 1.2, 7))
ax.set_ylabel(r"Predicted value, $\gamma^{*}/\gamma_{100}$")	    
ax.set_xlabel(r"Actual value, $\gamma^{*}/\gamma_{100}$") 
plt.subplots_adjust(left = 0.2, bottom = 0.15)
legend = plt.legend(loc = 'upper left')
plt.savefig("Sub100_gamma{Sub}_gamma{100}_Test.png", dpi=600)

mae_test = mean_absolute_error(y_test[:,0], y_pred_test[:,0])
mse_test = mean_squared_error(y_test[:,0], y_pred_test[:,0])
r2_test = r2_score(y_test[:,0], y_pred_test[:,0])
rmse_test = np.sqrt(mse_train)
print(r"--------------$\gamma^{*}/\gamma_{100}$--------------")
print('Test MAE:', mae_test)
print('Test MSE:', mse_test)
print('Test R^2:', r2_test)
print('Test RMSE:', rmse_test)

plt.show() 

################################################################################

# Evaluate the model with the training set
y_pred_train = model.predict(X_train)

# Ensure y_train and y_pred_train are of the correct data type
y_train = np.array(y_train).astype(np.float64)
y_pred_train = np.array(y_pred_train).astype(np.float64)

fig  = plt.figure(figsize=(4.0, 3.5)) # figsize=(4.0, 3.5)
ax = fig.add_subplot(1, 1, 1)
# print (len(y_train[:,0]), len(X_train[:,0]), len(y_test[:,0]), len(y_pred[:,0]))
plt.scatter(y_train[:,2], y_pred_train[:,2], s = 5.5, marker='*', color = (0.0, 0.79, 0.66), label = 'Training set') 
plt.plot([0.5, 1.8], [0.5, 1.8], 'k--', linewidth = 0.8)   
# plt.scatter(y_test[:,0], model.predict(X_test)[:,0], s = 1.5, color = (1.0, 0.0, 0.33), label = 'Test set')
ax.set_xlim([0.5, 1.8])
ax.set_ylim([0.5, 1.8])
ax.set_xticks(np.linspace(0.5,1.8,6))
ax.set_yticks(np.linspace(0.5,1.8,6))
ax.set_ylabel(r"Predicted value, $\gamma_{111}/\gamma_{100}$")	    
ax.set_xlabel(r"Actual value, $\gamma_{111}/\gamma_{100}$") 
plt.subplots_adjust(left = 0.18, bottom = 0.15)
legend = plt.legend(loc = 'upper left')
plt.savefig("Sub100_gamma{111}_gamma{100}_train.png", dpi=600)

mae_train = mean_absolute_error(y_train[:,2], y_pred_train[:,2])
mse_train = mean_squared_error(y_train[:,2], y_pred_train[:,2])
r2_train = r2_score(y_train[:,2], y_pred_train[:,2])
rmse_train = np.sqrt(mse_train)
print(r"--------------$\gamma^{111}/\gamma_{100}$_train--------------")
print('Train MAE:', mae_train)
print('Train MSE:', mse_train)
print('Train R^2:', r2_train)
print('Train RMSE:', rmse_train)

plt.show() 

################################################################################

fig  = plt.figure(figsize=(4.0, 3.5)) # figsize=(4.0, 3.5)
ax = fig.add_subplot(1, 1, 1)
plt.scatter(y_test[:,2], y_pred_test[:,2], s = 5.5, marker='>', color = (0.0, 0.79, 0.66), label = 'Test set') 
plt.plot([0.5, 1.8], [0.5, 1.8], 'k--', linewidth = 0.8)   
# plt.scatter(y_test[:,0], model.predict(X_test)[:,0], s = 1.5, color = (1.0, 0.0, 0.33), label = 'Test set')
ax.set_xlim([0.5, 1.8])
ax.set_ylim([0.5, 1.8])
ax.set_xticks(np.linspace(0.5,1.8,6))
ax.set_yticks(np.linspace(0.5,1.8,6))
ax.set_ylabel(r"Predicted value, $\gamma_{111}/\gamma_{100}$")	    
ax.set_xlabel(r"Actual value, $\gamma_{111}/\gamma_{100}$") 
plt.subplots_adjust(left = 0.18, bottom = 0.15)
legend = plt.legend(loc = 'upper left')
plt.savefig("Sub100_gamma{111}_gamma{100}_test.png", dpi=600)

mae_test = mean_absolute_error(y_test[:,2], y_pred_test[:,2])
mse_test = mean_squared_error(y_test[:,2], y_pred_test[:,2])
r2_test = r2_score(y_test[:,2], y_pred_test[:,2])
rmse_test = np.sqrt(mse_train)
print(r"--------------$\gamma_{111}/\gamma_{100}$--------------")
print('Test MAE:', mae_test)
print('Test MSE:', mse_test)
print('Test R^2:', r2_test)
print('Test RMSE:', rmse_test)

plt.show() 




