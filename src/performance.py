#Filename: performance.py

def assess_regressor_set(model, features, target, set_name):
	'''Input parameters:
	model=trained model 
	features=set of features 
	set = name of set: 
	'''
	predict = model.predict(features)
	#Calaculate RMSE and MAE 
	# Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(target, predictions))
    mae = mean_absolute_error(target, predictions)

    # Print the results
    print(f"{set_name} Set RMSE: {rmse:.4f}")
    print(f"{set_name} Set MAE: {mae:.4f}")
    return rmse, mae

def fit_assess_regressor(model, X_train, y_train, X_val, y_val):
    """
    Train the model on the training set and evaluate on both training and validation sets.

    Parameters:
    model: Regressor model to train
    X_train: Features for the training set
    y_train: Target for the training set
    X_val: Features for the validation set
    y_val: Target for the validation set

    Output: None
    """
    # Train the model
    model.fit(X_train, y_train)

    # Assess the model on the training set
    print("Training Set Performance:")
    assess_regressor_set(model, X_train, y_train, "Training")

    # Assess the model on the validation set
    print("\nValidation Set Performance:")
    assess_regressor_set(model, X_val, y_val, "Validation")