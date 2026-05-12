# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.
## Neural Network Model
<img width="998" height="698" alt="image" src="https://github.com/user-attachments/assets/a757df16-cd3e-4a0a-99c3-8ac7e249f2af" />

## DESIGN STEPS 
1. Load and Preprocess Data
2. Get the shape of the first image in the training dataset
3. Get the shape of the first image in the test dataset
4. Train the Model
5. Test the Model
6. Predict on a Single Image
7. Display the image  

## PROGRAM

### Name:DEEPAKK.C.R

### Register Number:212224040059

```python
class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
      x=self.pool((torch.relu(self.conv1(x))))
      x=self.pool((torch.relu(self.conv2(x))))
      x=self.pool((torch.relu(self.conv3(x))))
      x=x.view(x.size(0),-1)
      x=torch.relu(self.fc1(x))
      x=torch.relu(self.fc2(x))
      x=self.fc3(x)
      return x



# Initialize the Model, Loss Function, and Optimizer
model=CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):

     for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()


    # write your code here


    print('Name:DEEPAKKC.R        ')
    print('Register Number: 212224040059      ')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        
        
        
       

```

### OUTPUT

## Training Loss per Epoch

<img width="634" height="311" alt="image" src="https://github.com/user-attachments/assets/600ba2d4-64ed-4f73-b67b-34f4e2953aad" />

## Confusion Matrix
<img width="1097" height="760" alt="image" src="https://github.com/user-attachments/assets/46941356-3bd0-4d03-b7d7-77d8e4e50836" />

## Classification Report
<img width="659" height="382" alt="image" src="https://github.com/user-attachments/assets/8f435127-2223-450a-83ff-6034599d81f9" />

### New Sample Data Prediction
<img width="716" height="693" alt="image" src="https://github.com/user-attachments/assets/c0b26b94-f17e-4188-9cde-c9835d911243" />

## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
