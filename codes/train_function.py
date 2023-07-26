import torch
import time
from datetime import datetime
import PIL.Image as pil_image
import numpy as np
import tqdm
from codes.utils import *

def train_func(
    model,
    device,
    criterion,
    optimizer,
    scheduler,
    n_epochs,
    train_loader,
    valid_loader,
    save_path,
    scale,
    Train_loss=[],
    Valid_loss=[],
    valid_loss_min = np.Inf,
    valid_psnr_max = 0.,
    scheduler_flag=False,
    lvl=0,
    bicubic_flag=False,
    ):

  model.to(device)
  current_epoch=len(Train_loss)
  for epoch in range((current_epoch+1), n_epochs+current_epoch+1):
      train_loss = 0.0
      valid_loss = 0.0
      model.train()
      if (epoch-lvl) == 100: break
      print('Current epoch:',epoch,' Current lr:',get_lr(optimizer),' last save at epoch:',lvl,' Max Valid PSNR:',valid_psnr_max)
      ###################
      # train the model #
      ###################
      # model by default is set to train

      start_time = time.time()
      # for batch_i, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Epoch progress'):
      for batch_i, (data,target) in enumerate(train_loader):

          data.requires_grad = True
          data, target = data.to(device), target.to(device)

          optimizer.zero_grad()# clear the gradients of all optimized variables
          output = model(data)# forward pass: compute predicted outputs by passing inputs to the model
          loss = criterion(output, target)# calculate the loss
          loss.backward()# backward pass: compute gradient of the loss with respect to model parameters
          # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.4) # Gradient Clipping
          optimizer.step()# perform a single optimization step (parameter update)
          train_loss += loss.item()
      ######################
      # validate the model #
      ######################
      model.eval()
      with torch.no_grad():
        for batch_idx, (data,target) in enumerate(valid_loader):
          data, target = data.cuda(), target.cuda()
          output = model(data)
          loss = torch.nn.MSELoss()(output, target)# calculate the batch loss
          valid_loss += loss.item()

      # calculate average losses
      train_loss = train_loss/len(train_loader)
      valid_loss = valid_loss/len(valid_loader)
      valid_psnr = psnr(valid_loss)
      # print training/validation statistics
      print('\nTraining Loss:   {:.16f}'.format(train_loss))
      print('Validation Loss: {:.16f}'.format(valid_loss))
      print('Validation PSNR: {:.16f}'.format(valid_psnr))
      Train_loss.append(train_loss)
      Valid_loss.append(valid_loss)
      if scheduler_flag:
        scheduler.step(valid_loss)
      if valid_loss < valid_loss_min:
        print('\033[1;35;40mSaving model ...\033[0;0m')
        lvl=epoch
        valid_loss_min = valid_loss
        valid_psnr_max = valid_psnr
        checkpoint = {'optimizer': optimizer,
                      'optimizer state': optimizer.state_dict(),
                      'state_dict': model.module.state_dict(),
                      'scheduler state': scheduler.state_dict(),
                      'Train_loss':Train_loss,
                      'Valid_loss':Valid_loss,
                      'minimum validation loss':valid_loss_min}
        torch.save(checkpoint, save_path)
      end_time = time.time()
      diff=end_time-start_time
      print('Time spent: {0}:{1} mins'.format(str(int(diff-(diff%60))//60),str(int(diff%60)).zfill(2)))
      print("End time:", str(datetime.fromtimestamp(datetime.timestamp(datetime.now())//1))[11:],'\n')
      checkpoint = {'optimizer': optimizer,
                      'optimizer state': optimizer.state_dict(),
                      'state_dict': model.module.state_dict(),
                      'scheduler state': scheduler.state_dict(),
                      'Train_loss':Train_loss,
                      'Valid_loss':Valid_loss,
                      'minimum validation loss':valid_loss_min}
      torch.save(checkpoint, save_path[:-4]+'_LAST.pth')
