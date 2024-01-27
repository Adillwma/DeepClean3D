### CUSTOM LOSS FUNCTION TESTER SCRIPT ###

import torch
import torch.nn as nn
import torch.optim as optim
import timeit
import matplotlib.pyplot as plt

# Define a simple neural network for testing
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc(x)
        return x

def test_custom_loss(*args):

    ### Preparation ###
    manual_seed = 10
    torch.manual_seed(manual_seed)

    # Generate dummy data
    input_data = torch.randn(1, 1, 100, 10)
    target_data = torch.randn(1, 1, 100, 1)

    # Normalize the data
    input_data = input_data / input_data.max()
    target_data = target_data / target_data.max()

    # Initialize the network and loss function
    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    ### Testing Routine ###
    all_loss_values = []
    loss_labels = []

    for loss_class in args:
        torch.manual_seed(manual_seed+1)
        
        criterion = loss_class 
        print("Testing loss function:", type(criterion).__name__)
        
        # Test code runs without errors and check if can work with different common input shapes
        
        # Test with tensors of shape (x)
        try:
            loss_1 = criterion(input_data[0,0,0], input_data[0,0,0])
            #print("Loss for shape (x):", loss_1)
        except Exception as e:
            print("Loss function cannot work with shape (x) Check loss fn code for error.\n Failed wil error message:\n", str(e))

        # Test with tensors of shape (x, y)
        try:
            loss_2 = criterion(input_data[0,0], input_data[0,0])
            #print("Loss for shape (x, y):", loss_2)
        except Exception as e:
            print("Loss function cannot work with shape (x, y) Check loss fn code for error.\n Failed wil error message:\n", str(e))

        # Test with tensors of shape (channels, x, y)
        try:
            loss_3 = criterion(input_data[0], input_data[0])
            #print("Loss for shape (channels, x, y):", loss_3)
        except Exception as e:
            print("Loss function cannot work with shape (channel, x, y) Check loss fn code for error.\n Failed wil error message:\n", str(e))
            
        # Test with tensors of shape (batches, channels, x, y)
        try:
            loss_4 = criterion(input_data, input_data)
            #print("Loss for shape (batches, channels, x, y):", loss_4)
        except Exception as e:
            print("Loss function cannot work with shape (batch, channel, x, y) Check loss fn code for error.\n Failed wil error message:\n", str(e))
            break

        # Test speed with timeit
        def time_custom_loss():
            output = net(input_data)
            loss = criterion(output, target_data)

        print("Time taken to compute loss:")
        print(min(timeit.repeat(time_custom_loss, repeat=10, number=1000)))



        # Test gradient computation
        try:
            for _ in range(100):
                optimizer.zero_grad()
                output = net(input_data)
                loss = criterion(output, target_data)
                loss.backward()
                optimizer.step()
            print("Gradient computation successful!")
        except:
            print("Gradient computation failed!")
        """
        # Create a plot of loss vs. difference level
        x = 10
        y = 10
        diff_levels = torch.linspace(-1, 1, x*y*2)
        loss_values = []
        dummy_img1 = torch.zeros(1,1,x,y)
        #set 50% of pixels to hits
        dummy_img1[:,:,0:5,:] = 0.5
        dummy_img2 = torch.zeros(1,1,x,y)
        for rep in range(2):
            for pixel in range(x*y):
                dummy_img2[0,0,pixel%x,pixel//x] = dummy_img2[0,0,pixel%x,pixel//x] + 0.5
                #plt.imshow(dummy_img2[0,0])
                #plt.show()
                #plt.imshow(dummy_img1[0,0])
                #plt.show()
                loss = criterion(dummy_img1, dummy_img2)
                loss_values.append(loss.item())
        loss_labels.append(type(loss_class).__name__)
        all_loss_values.append(loss_values)
        """
        # Create a plot of loss vs. difference level
        dummy_img1 = torch.randn(1,1,10,10)
        diff_levels = torch.linspace(-1, 1, 100)
        loss_values = []
        for diff_level in diff_levels:
            dummy_img1 = torch.randn(1,1,10,10)
            dummy_img2 = dummy_img1 + diff_level
            loss = criterion(dummy_img1, dummy_img2)
            loss_values.append(loss.item())
        loss_labels.append(type(loss_class).__name__)
        all_loss_values.append(loss_values)



    for lossvals, class_label in zip(all_loss_values, loss_labels):
        plt.plot(diff_levels, lossvals, label=class_label)
    plt.xlabel("Difference Level")
    plt.ylabel("Custom Loss")
    plt.title("Loss vs. Difference Level")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()