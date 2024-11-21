import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Reshape

def get_model(input_size, num_classes):
	# Get the input size and number of classes

	# Define the CNN model
	model = Sequential([
	    # Reshape the input for Conv1D
	    Reshape((input_size, 1), input_shape=(input_size,)),  
	    
	    # Add Conv1D layers
	    Conv1D(filters=32, kernel_size=3, activation='relu'),
	    Conv1D(filters=64, kernel_size=3, activation='relu'),
	    
	    # Flatten the output
	    Flatten(),
	    
	    # Fully connected layers
	    Dense(128, activation='relu'),
	    Dropout(0.5),  # Dropout to prevent overfitting
	    Dense(num_classes, activation='softmax')  # Output layer with softmax for classification
	])

	# Compile the model
	model.compile(optimizer='sgd',
	              loss='categorical_crossentropy',
	              metrics=['accuracy'])

	return model
