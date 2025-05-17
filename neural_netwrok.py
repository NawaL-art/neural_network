import numpy as np

# Simple Neural Network for Salary Prediction
class NeuralNetwork:
    def __init__(self, input_size):
        # Initialize weights and bias randomly
        # This corresponds to the connections between input layer and output layer
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)[0]
        
    def sigmoid(self, x):
        # Activation function - transforms the input to a value between 0 and 1
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function for backpropagation
        return x * (1 - x)
    
    def feed_forward(self, inputs):
        # Calculate the weighted sum of inputs plus bias
        # This is like the surfing example: (x1*w1 + x2*w2 + x3*w3 - threshold)
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        
        # Apply the activation function
        output = self.sigmoid(weighted_sum)
        return output
    
    def train(self, training_inputs, training_outputs, iterations):
        for iteration in range(iterations):
            # For each training example
            for inputs, expected_output in zip(training_inputs, training_outputs):
                # Step 1: Feed forward to get the current prediction
                prediction = self.feed_forward(inputs)
                
                # Step 2: Calculate the error
                error = expected_output - prediction
                
                # Step 3: Backpropagation - update weights and bias
                adjustment = error * self.sigmoid_derivative(prediction)
                self.weights += inputs * adjustment
                self.bias += adjustment
    
    def predict(self, inputs):
        # Make a prediction based on the input
        return self.feed_forward(inputs)


# Example usage: Salary prediction based on years of experience, education level, and skills
if __name__ == "__main__":
    # Create a neural network with 3 input features
    nn = NeuralNetwork(input_size=3)
    
    # Training data: [years_experience, education_level(1-5), skills(1-10)]
    # Each row is one training example
    training_inputs = np.array([
        [1, 2, 3],    # Junior with minimal experience
        [3, 3, 5],    # Mid-level
        [5, 4, 7],    # Senior
        [10, 5, 9],   # Expert
    ])
    
    # Normalized target salaries (scaled between 0 and 1)
    # These would be something like: $50K, $70K, $90K, $120K
    training_outputs = np.array([0.2, 0.4, 0.6, 0.8])
    
    # Train the neural network
    print("Training the neural network...")
    nn.train(training_inputs, training_outputs, 10000)
    
    # Test with new data
    new_employee = np.array([2, 3, 4])  # 2 years experience, bachelor's degree, decent skills
    predicted_salary = nn.predict(new_employee)
    
    # Convert the normalized prediction back to an actual salary (assuming $150K is max)
    actual_salary = predicted_salary * 150000
    
    print(f"Input features: {new_employee}")
    print(f"Predicted normalized salary: {predicted_salary:.4f}")
    print(f"Predicted actual salary: ${actual_salary:.2f}")
    
    # Let's try another example
    experienced_employee = np.array([8, 4, 8])
    predicted_salary = nn.predict(experienced_employee)
    actual_salary = predicted_salary * 150000
    
    print(f"\nInput features: {experienced_employee}")
    print(f"Predicted normalized salary: {predicted_salary:.4f}")
    print(f"Predicted actual salary: ${actual_salary:.2f}")
