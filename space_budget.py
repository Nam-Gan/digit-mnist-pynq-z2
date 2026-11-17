in_depth = 8
activation_depth = 8
weight_depth = 8
bias_depth = 16
num_layers = 3
layer0_neurons = 784 
layer1_neurons = 128
layer2_neurons = 64 
layer3_neurons = 10

space_bits = (in_depth * layer0_neurons) + (weight_depth*layer0_neurons*layer1_neurons + bias_depth*layer1_neurons) + (weight_depth*layer1_neurons*layer2_neurons + bias_depth*layer2_neurons) + (weight_depth*layer2_neurons*layer3_neurons + bias_depth*layer3_neurons)
print("Total space in bits: ", space_bits)
print("Bytes: ", space_bits/8);

