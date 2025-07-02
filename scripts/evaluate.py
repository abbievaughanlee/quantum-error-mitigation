
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# params
MODEL_FILE = 'models/model_2q_d10_01.pth'
TEST_DATA_FILE = 'data/testing_dataset_2q_d1001.npz'
NUM_SAMPLES_TO_PLOT = 3 # How many example plots to generate

# redefine the model
class ErrorMitigationNet(nn.Module):
    def __init__(self, num_features):
        super(ErrorMitigationNet, self).__init__()
        self.layer1 = nn.Linear(num_features, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_features)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.softmax(self.layer3(x), dim=1)
        return x

# fidelity: classical fidelity between two probability distributions
def fidelity(p, q):
    return (np.sqrt(p * q).sum())**2


if __name__ == "__main__":
    # load the test data
    print(f"Loading test data from {TEST_DATA_FILE}...")
    test_data = np.load(TEST_DATA_FILE)
    X_test_np = test_data['x']
    y_test_np = test_data['y']
    
    # conv to PyTorch tensors
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    
    # load the trained model
    num_features = X_test.shape[1]
    model = ErrorMitigationNet(num_features)
    print(f"Loading trained model from {MODEL_FILE}...")
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()  

    # make predictions on the test set
    print("Making predictions on the test set...")
    with torch.no_grad():
        y_pred = model(X_test)
    
    # conv predictions back to numpy for calculations
    y_pred_np = y_pred.numpy()
    
    # calculate fidelities
    fidelities_noisy = []
    fidelities_corrected = []

    for i in range(len(X_test_np)):
        noisy_vec = X_test_np[i]
        ideal_vec = y_test_np[i]
        corrected_vec = y_pred_np[i]

        fidelities_noisy.append(fidelity(noisy_vec, ideal_vec))
        fidelities_corrected.append(fidelity(corrected_vec, ideal_vec))
        
    avg_noisy_fidelity = np.mean(fidelities_noisy)
    avg_corrected_fidelity = np.mean(fidelities_corrected)

    print("\n--- Evaluation Results ---")
    print(f"Average Fidelity of Noisy Data vs. Ideal:     {avg_noisy_fidelity:.4f}")
    print(f"Average Fidelity of Corrected Data vs. Ideal: {avg_corrected_fidelity:.4f}")
    print("--------------------------")
    
    #plot some examples
    print(f"Generating {NUM_SAMPLES_TO_PLOT} comparison plots...")
    for i in range(NUM_SAMPLES_TO_PLOT):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        bar_width = 0.25
        index = np.arange(num_features)
        
        ax.bar(index - bar_width, y_test_np[i], bar_width, label='Ideal (True)', color='green')
        ax.bar(index, X_test_np[i], bar_width, label=f'Noisy (Fid: {fidelities_noisy[i]:.3f})', color='red')
        ax.bar(index + bar_width, y_pred_np[i], bar_width, label=f'ML Corrected (Fid: {fidelities_corrected[i]:.3f})', color='blue')
        
        ax.set_xlabel('Measurement Outcome State')
        ax.set_ylabel('Probability')
        ax.set_title(f'Sample {i+1}: Comparison of Distributions')
        ax.set_xticks(index)
        ax.set_xticklabels([f'{j:0{num_features//2}b}' for j in range(num_features)])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'figures/2q_d10_01_evaluation_samples/evaluation_sample_{i+1}.png')
        
    print(f"Plots saved in 'figures/' directory.")