# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET

Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

## Neural Network Model
<img width="998" height="698" alt="image" src="https://github.com/user-attachments/assets/a757df16-cd3e-4a0a-99c3-8ac7e249f2af" />

## DESIGN STEPS
## STEP 1:

Collect and preprocess the image dataset.

## STEP 2:

Import required deep learning libraries.

## STEP 3:

Build the CNN architecture.

## STEP 4:

Train the CNN model using training data.

## STEP 5:

Evaluate the model performance using test data.

## STEP 6:

Test the model with new images and verify predictions.





## PROGRAM

### Name: C.R.DEEPAKK

### Register Number: 212224040059

```python
class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = self.fc3(x)

        return x



# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):

    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # tqdm adds a progress bar to show how long it will take
        loop = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')

        for images, labels in loop:
            # MOVE DATA TO GPU HERE
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f'\nName: C.R.DEEPAKK')
        print(f'Register Number: 212224040059')
        print(f'Finished Epoch {epoch+1}, Avg Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

<img width="777" height="323" alt="image" src="https://github.com/user-attachments/assets/66efda09-971f-4b13-9f9c-8307ba38d0ac" />


## Confusion Matrix

<img width="892" height="862" alt="image" src="https://github.com/user-attachments/assets/781475aa-1be4-4604-a4ab-72e4b23eaa06" />


## Classification Report
<img width="551" height="402" alt="image" src="https://github.com/user-attachments/assets/57c672bd-f14c-455d-8161-cde811b4a7a4" />


### New Sample Data Prediction
<img width="572" height="661" alt="image" src="https://github.com/user-attachments/assets/fc9c288b-d704-4130-81d6-d787933d54ae" />


## RESULT
The CNN model was successfully trained and tested, achieving accurate image classification for new input images.
