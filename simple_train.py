import new_dataset
from nets.SqueezeNet import SqueezeNet
from torch.autograd import Variable
import torch.nn.utils as nnutils
import torch

def main():

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    with torch.cuda.device(dev): 
        net = SqueezeNet # .to(device)
        criterion = torch.nn.MSELoss() # .to(device)
        optimizer.torch.optim.Adadelta(net.parameters()) # params are weights and biases

        net.train()

        train_dataset = Dataset('/home/nitzan/2-25-19_with_mic/train_data', 6)
        train_dataloader = torch.utils.data.Dataloader(train_dataset, batch_size=100, shuffle=False)

        for epoch in range(2):
            print('Starting training epoch #{}'.format(epoch))
            running_loss = 0.0

            for batch_idx, (camera, truth) in enumerate(train_dataloader):
                camera = camera #.cuda()
                truth = truth #.cuda()

                optimizer.zero_grad()
                outputs = net(Variable(camera)) #.cuda()
                loss = criterion(outputs, Variable(truth))

                #Back prop
                loss.backward()
                optimizer.step()

                # print stats
                running_loss += loss.item()
                if batch_idx % 10 == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0




        print('Finished training!')


        # testing to be continued..