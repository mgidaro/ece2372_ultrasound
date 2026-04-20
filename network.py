import os
import platform
import ctypes

#ctypes.CDLL(os.path.normpath("C:\\Users\\Mike\\AppData\\Local\\Programs\\Python\\Python314\\Lib\\site-packages\\torch\\lib\\c10.dll"))
#if platform.system() == "Windows":
#    import ctypes
#    from importlib.util import find_spec
#    try:
#        if (spec := find_spec("torch")) and spec.origin and os.path.exists(
#            dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
#        ):
#            ctypes.CDLL(os.path.normpath(dll_path))
#    except Exception:
#        pass

from typing import Tuple

import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from data_loading import RegressionTaskData


class CNNRegression(nn.Module):
    #Need to add layers to match VGG-11
    def __init__(self, image_size: Tuple[int, int, int], m_num: int):
        super(CNNRegression, self).__init__()
        self.image_size = image_size
        if m_num == 0:
        
            self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
            #self.batch1 = nn.BatchNorm2d(num_features=4)
            self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
            #self.batch2 = nn.BatchNorm2d(num_features=16)
            
            self.linear_line_size = int(16*(image_size[1]//4)*(image_size[2]//4))
            self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)    
            self.fc2 = nn.Linear(in_features=128, out_features=1)

        elif m_num == 1:
            self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
            
            self.linear_line_size = int(16*(image_size[1]//8)*(image_size[2]//8))
            self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)    
            self.fc2 = nn.Linear(in_features=128, out_features=1)

        elif m_num == 2:
            self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=16, kernel_size=3, stride=1, padding=1)
            
            self.linear_line_size = int(16*(image_size[1]//2)*(image_size[2]//2))
            self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)    
            self.fc2 = nn.Linear(in_features=128, out_features=1)

        elif m_num == 3:
            self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=2, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1)
            
            self.linear_line_size = int(8*(image_size[1]//4)*(image_size[2]//4))
            self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)    
            self.fc2 = nn.Linear(in_features=128, out_features=1)

        elif m_num == 4:
            self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=8, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
            
            self.linear_line_size = int(32*(image_size[1]//4)*(image_size[2]//4))
            self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)    
            self.fc2 = nn.Linear(in_features=128, out_features=1)

        elif m_num == 5:
            self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
            
            self.linear_line_size = int(16*(image_size[1]//4)*(image_size[2]//4))
            self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=1)    

        elif m_num == 6:
            self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
            
            self.linear_line_size = int(16*(image_size[1]//4)*(image_size[2]//4))
            self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=1024)
            self.fc2 = nn.Linear(in_features=1024, out_features=256)
            self.fc3 = nn.Linear(in_features=256, out_features=1)

        elif m_num == 7:
            self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
            self.batch1 = nn.BatchNorm2d(num_features=4)
            self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.batch2 = nn.BatchNorm2d(num_features=16)
            
            self.linear_line_size = int(16*(image_size[1]//4)*(image_size[2]//4))
            self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)    
            self.batch3 = nn.BatchNorm1d(num_features=128)
            self.fc2 = nn.Linear(in_features=128, out_features=1)

        elif m_num == 8:
            self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
            
            self.linear_line_size = int(16*(image_size[1]//4)*(image_size[2]//4))
            self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)    
            self.fc2 = nn.Linear(in_features=128, out_features=1)

        elif m_num == 9:
            self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
            
            self.linear_line_size = int(16*(image_size[1]//4)*(image_size[2]//4))
            self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=1024)
            self.fc2 = nn.Linear(in_features=1024, out_features=256)
            self.fc3 = nn.Linear(in_features=256, out_features=64)
            self.fc4 = nn.Linear(in_features=64, out_features=1)

        else:
            raise NameError('model doesnt exist!')
        
    def forward(self, x):
        """
        Passes the data through the network.
        There are commented out print statements that can be used to 
        check the size of the tensor at each layer. These are very useful when
        the image size changes and you want to check that the network layers are 
        still the correct shape.
        """
        if m_num == 0:
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


            x = x.view(-1, self.linear_line_size)
            # print(f'view1 {x.size()}')
            #x = torch.flatten(x,1)
            x = self.fc1(x)
            # print(f'fc1 {x.size()}')
            x = nn.functional.relu(x)
            # print(f'relu2 {x.size()}')
            #x = nn.functional.dropout(x, p=0.5, inplace=False)
            x = self.fc2(x)

        elif m_num == 1:
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv3(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)


            x = x.view(-1, self.linear_line_size)
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)

        elif m_num == 2:
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            
            x = x.view(-1, self.linear_line_size)
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)

        elif m_num == 3:
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            
            x = x.view(-1, self.linear_line_size)
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)

        elif m_num == 4:
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            
            x = x.view(-1, self.linear_line_size)
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)

        elif m_num == 5:
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            
            x = x.view(-1, self.linear_line_size)
            x = self.fc1(x)

        elif m_num == 6:
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            
            x = x.view(-1, self.linear_line_size)
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)
            x = nn.functional.relu(x)
            x = self.fc3(x)

        elif m_num == 7:
            x = self.conv1(x)
            x = self.batch1(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv2(x)
            x = self.batch2(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            
            x = x.view(-1, self.linear_line_size)
            x = self.fc1(x)
            x = self.batch3(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)

        elif m_num == 8:
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            
            x = x.view(-1, self.linear_line_size)
            x = nn.functional.dropout(x, p=0.2, inplace=False)
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)

        elif m_num == 9:
            x = self.conv1(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv2(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
            
            x = x.view(-1, self.linear_line_size)
            x = self.fc1(x)
            x = nn.functional.relu(x)
            x = self.fc2(x)
            x = nn.functional.relu(x)
            x = self.fc3(x)
            x = nn.functional.relu(x)
            x = self.fc4(x)

        return x
    

def train_network(device, n_epochs: int, image_size: Tuple[int, int, int], image_folder_path: str, m_num: int):
    """
    This trains the network for a set number of epochs.
    """
    #if image_size[0] == 1: #MRG: this should always be grayscale
    #grayscale = True
    #else:
    #    grayscale = False
    assert image_size[1] == 280, 'size 1 wasnt 280'
    assert image_size[2] == 150, 'size 2 wasnt 150'
    print(f'training')
    #resize_size = image_size[1]
    regression_task = RegressionTaskData(grayscale=True, image_folder_path=image_folder_path)#, resize_size=resize_size)

    # Define the model, loss function, and optimizer
    model = CNNRegression(image_size=image_size, m_num=m_num)
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


def load_model(image_size, filename, m_num):
    """
    Load the model from the saved state dictionary.
    """
    model = CNNRegression(image_size, m_num=m_num)
    model.load_state_dict(torch.load(filename))
    return model


def evaluate_network(model, device, image_size: Tuple[int, int, int] = (3, 640, 470), image_folder_path: str = "C:/Users/Mike/Documents/imgAnalysis/recordings/US_frames/US_version6"):
    """
    This evaluates the network on the test data.
    """
    #if image_size[0] == 1:
    #    grayscale = True
    #else:
    #    grayscale = False
    assert image_size[1] == 280, 'size 1 wasnt 280'
    assert image_size[2] == 150, 'size 2 wasnt 150'
    #resize_size = image_size[1]
    regression_task = RegressionTaskData(grayscale=True, image_folder_path=image_folder_path)#, resize_size=resize_size)
    criterion = nn.MSELoss()
    model.eval()

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
            #print(angle_error)
            #print(target_angles)
            #print(np.abs(target_angles - output_angles))
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
    
    m_num = 9
    num_epochs = 20
    #dont forget
    filename = f'C:/Users/Mike/Documents/imgAnalysis/ece2372_ultrasound/models/m{m_num}.pth' 
    image_folder_path = "C:/Users/Mike/Documents/imgAnalysis/ece2372_ultrasound/US_version6"
    image_size: Tuple[int, int, int] = (1, 280, 150)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    

    
    # Train the model
    model = train_network(device, n_epochs=num_epochs, image_size=image_size, image_folder_path=image_folder_path, m_num=m_num)

    # Save the model
    save_model(model, filename=filename)

    # Load the model
    model = load_model(image_size=image_size, filename=filename, m_num=m_num)
    model.to(device)

    '''print(model)'''

    # Evaluate the model
    evaluate_network(model, device, image_size=image_size, image_folder_path=image_folder_path)
    