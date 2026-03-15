
from typing import Tuple

import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from data_loading import RegressionTaskData


class CNNRegression(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int] = (3, 640, 470)):
        super(CNNRegression, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
        #self.batch1 = nn.BatchNorm2d(num_features=4)
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        #self.batch2 = nn.BatchNorm2d(num_features=16)
        #self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        #self.batch3 = nn.BatchNorm2d(num_features=16)
        #self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        #self.batch4 = nn.BatchNorm2d(num_features=16)
        ##self.pool4  = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        #self.batch5 = nn.BatchNorm2d(num_features=32)
        #self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        ##self.batch6 = nn.BatchNorm2d(num_features=32)
        ##self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        ##self.batch7 = nn.BatchNorm2d(num_features=32)
        #self.conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        #self.batch8 = nn.BatchNorm2d(num_features=32)
        #self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #self.avg = nn.AdaptiveAvgPool2d(output_size=7)
        
        self.linear_line_size = int(16*(image_size[1]//4)*(image_size[2]//4))
        self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)    
        #self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=1)

        
    def forward(self, x):
        """
        Passes the data through the network.
        There are commented out print statements that can be used to 
        check the size of the tensor at each layer. These are very useful when
        the image size changes and you want to check that the network layers are 
        still the correct shape.
        """

        x = self.conv1(x)
        # print('Size of tensor after each layer')
        # print(f'conv1 {x.size()}')
        #x = self.batch1(x)
        # print(f'batch1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu1 {x.size()}')
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # print(f'pool1 {x.size()}')
        x = self.conv2(x)
        # print(f'conv2 {x.size()}')
        #x = self.batch2(x)
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # print(f'pool2 {x.size()}')
        #x = self.conv3(x)
        #x = self.batch3(x)
        #x = nn.functional.relu(x)
        #x = self.conv4(x)
        #x = self.batch4(x)
        #x = nn.functional.relu(x)
        #x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        #x = self.conv5(x)
        #x = self.batch5(x)
        #x = nn.functional.relu(x)
        #x = self.conv6(x)
        #x = self.batch6(x)
        #x = nn.functional.relu(x)
        #x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        #x = self.conv7(x)
        ##x = self.batch7(x)
        #x = nn.functional.relu(x)
        #x = self.conv8(x)
        ##x = self.batch8(x)
        #x = nn.functional.relu(x)
        #x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        #x = self.avg(x)

        x = x.view(-1, self.linear_line_size)
        # print(f'view1 {x.size()}')
        #x = torch.flatten(x,1)
        x = self.fc1(x)
        # print(f'fc1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        #x = nn.functional.dropout(x, p=0.5, inplace=False)
        #x = self.fc2(x)
        # print(f'fc2 {x.size()}')
        #x = nn.functional.relu(x)
        #x = nn.functional.dropout(x, p=0.5, inplace=False)
        x = self.fc3(x)

        return x
    

def train_network(device, n_epochs: int = 10, image_size: Tuple[int, int, int] = (3, 640, 470)):
    """
    This trains the network for a set number of epochs.
    """
    #if image_size[0] == 1: #MRG: this should always be grayscale
    #grayscale = True
    #else:
    #    grayscale = False
    assert image_size[1] == 640, 'size 1 wasnt 640'
    assert image_size[2] == 470, 'size 2 wasnt 470'
    #resize_size = image_size[1]
    regression_task = RegressionTaskData(grayscale=False, image_folder_path="C:/Users/Mike/Documents/imgAnalysis/recordings/US_frames/US_version5")#, resize_size=resize_size)

    # Define the model, loss function, and optimizer
    model = CNNRegression(image_size=image_size)
    model.to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    writer = SummaryWriter()
    for epoch in range(n_epochs):
        for i, (inputs, targets) in enumerate(regression_task.trainloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train Loss', loss.item(), i)

            # Print training statistics
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(regression_task.trainloader)}], Loss: {loss.item():.4f}')
    writer.close()

    return model


def save_model(model, filename='3_100_100.pth'):
    """
    After training the model, save it so we can use it later.
    """
    torch.save(model.state_dict(), filename)


def load_model(image_size=(3, 640, 470), filename='3_100_100.pth'):
    """
    Load the model from the saved state dictionary.
    """
    model = CNNRegression(image_size)
    model.load_state_dict(torch.load(filename))
    return model


def evaluate_network(model, device, image_size: Tuple[int, int, int] = (3, 640, 470)):
    """
    This evaluates the network on the test data.
    """
    #if image_size[0] == 1:
    #    grayscale = True
    #else:
    #    grayscale = False
    assert image_size[1] == 640, 'eval: size 1 wasnt 640'
    assert image_size[2] == 470, 'eval: size 2 wasnt 470'
    #resize_size = image_size[1]
    regression_task = RegressionTaskData(grayscale=False, image_folder_path="C:/Users/Mike/Documents/imgAnalysis/recordings/US_frames/US_version5")#, resize_size=resize_size)
    criterion = nn.MSELoss()

    # Evaluate the model on the test data
    with torch.no_grad():
        total_loss = 0
        total_angle_error = 0
        n_samples_total = 0
        for inputs, targets in regression_task.testloader:
            # Calculate the loss with the criterion we used in training
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            total_loss += loss.item()

            # We are actually predicting angles so we can calculate the angle error too
            # which is probably more meaningful to humans than the MSE loss
            output_angles = outputs.cpu().numpy()
            target_angles = targets.cpu().numpy()

            #print(output_angles)
            plt.scatter(target_angles,output_angles,c="blue")
            
            # calculate the angle error 
            angle_error = np.sum(np.abs(target_angles - output_angles)) #MRG: used to say np.rad2deg(target_angles...
            total_angle_error += angle_error
            n_samples_total += len(output_angles)

        mean_loss = total_loss / len(regression_task.testloader)
        mean_angle_error = total_angle_error / n_samples_total
        print(f'Test Loss: {mean_loss:.4f}')
        print(f'Test mean angle error: {mean_angle_error:.4f} degrees')

        
        plt.title("Model accuracy")
        plt.xlabel("Inputs (Ground Truth)")
        plt.ylabel("Outputs")
        plt.show()



if __name__ == '__main__':
    
    num_epochs = 20
    #dont forget
    filename = 'C:/Users/Mike/Documents/imgAnalysis/models/threeDtutorial.pth'
    image_size: Tuple[int, int, int] = (3, 640, 470)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    
    # Train the model
    #model = train_network(device, n_epochs=num_epochs, image_size=image_size)

    # Save the model
    #filename = f'{image_size[0]}_{image_size[1]}_{image_size[2]}.pth'
    #save_model(model, filename=filename)

    # Load the model
    model = load_model(image_size=image_size, filename=filename)
    model.to(device)

    '''print(model)'''

    # Evaluate the model
    evaluate_network(model, device, image_size=image_size)
    