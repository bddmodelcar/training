from new_dataset import Dataset
from nets.SqueezeNet10 import SqueezeNet
from torch.autograd import Variable
import torch.nn.utils as nnutils
import torch

def main():
    print('Begin training code')
    dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    start_epoch = 13
    save_dir = '/home/nitzan/2-25-19_with_mic/saves/2_saves/tests_epoch{}.weights'.format(start_epoch)

    
    # to truly continue training, I must also save the epoch, optimizer state_dict, and loss.

    net = SqueezeNet().to(dev)
    net.load_state_dict(torch.load(save_dir))
    criterion = torch.nn.MSELoss().to(dev)
    optimizer = torch.optim.Adadelta(net.parameters()) # params are weights and biases

    train_dataset = Dataset('/home/nitzan/2-25-19_with_mic/train_data', 10)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    test_dataset = Dataset('/home/nitzan/2-25-19_with_mic/test_data', 10)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    # TRAINING
    for epoch in range(start_epoch, start_epoch + 10):
        print('Starting training epoch #{}'.format(epoch))
        running_loss = 0.0
        net.train()

        for batch_idx, (camera, truth) in enumerate(train_dataloader, 1):
            camera = camera.to(dev)
            truth = truth.to(dev)

            optimizer.zero_grad()
            outputs = net(Variable(camera)).to(dev)
            loss = criterion(outputs, Variable(truth))

            #Back prop
            loss.backward()
            optimizer.step()

            # print stats
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch, batch_idx, running_loss / 10))
                running_loss = 0.0




    #print('Finished training!')
    #print('Starting testing...')
    
        running_test_loss = 0.0
    
        # VALIDATION
        for batch_idx, (camera, truth) in enumerate(test_dataloader, 1):
            camera = camera.to(dev)
            truth = truth.to(dev)

            # Forward
            net.eval()
            optimizer.zero_grad()
            outputs = net(camera)
            loss = criterion(outputs, truth)

            running_test_loss += loss.item()
        #if batch_idx % 1 == 0:
        print('[%d, %5d] loss: %.3f' % (epoch, batch_idx, running_test_loss / 8))
        #    running_loss = 0.0



        torch.save(net.state_dict(), '/home/nitzan/2-25-19_with_mic/saves/2_saves/tests_epoch' + str(epoch) + '.weights')
        print('saved', str(epoch))



if __name__ == '__main__':
    main()
