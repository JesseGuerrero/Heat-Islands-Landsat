#%%
'''
Get January 1991 -> 2024
LST
NDBI, NDVI, NDWI, NDBaI
DEM
'''
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

# Define the neural network class
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 2)
        self.fc3 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Instantiate the model
model = ANNModel()

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Define the optimizer (Adam optimizer)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Sample input data
X = torch.tensor([
    [26.4, 25.8, 25.0, 0.3, 0.2, 0.1, 0.05, 10.0],
    [28.1, 27.5, 26.9, 0.35, 0.25, 0.15, 0.08, 12.0],
    [29.5, 28.9, 28.3, 0.4, 0.3, 0.2, 0.1, 15.0]
], dtype=torch.float32)

# Corresponding LST predictions (ground truth)
y = torch.tensor([[27.0], [28.5], [30.0]], dtype=torch.float32)

# MLflow Tracking
with mlflow.start_run():
    # Log model summary
    print(model)

    # Log parameters
    mlflow.log_param("epochs", 5000)
    mlflow.log_param("learning_rate", 0.01)

    # Training the model
    epochs = 5000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # Log metrics every 500 epochs
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            mlflow.log_metric("loss", loss.item(), step=epoch + 1)

    # Save the model to MLflow
    mlflow.pytorch.log_model(model, "model")

    # Test the model
    test_data = torch.tensor([[28.0, 27.4, 26.8, 0.32, 0.22, 0.12, 0.06, 11.0]], dtype=torch.float32)
    predicted_LST = model(test_data).item()
    print(f"Predicted LST: {predicted_LST:.2f}°C")
    mlflow.log_metric("test_predicted_LST", predicted_LST)

    # Log the code as an artifact
    with open("experiment_code.py", "w") as f:
        f.write('''Your original code here''')
    mlflow.log_artifact("experiment_code.py")
