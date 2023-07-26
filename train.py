# importing libraries
import torch
import random
from codes.dataloading import *
from codes.train_function import *
from models_archs import *
import os

seed = 1993
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)

# check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


scale = 4#int(input('\nScale: '))
C = 32#int(input('C:'))
d = 1#int(input('d:'))
model = ESC_PAN(scale,C,d).to(device)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
save_path= "./saved_models/ESC_PAN_"+str(model.module.C)+"_"+str(model.module.d)+"/testx"+str(model.module.scale_factor)+".pth"
print("Model Parameters: {:,}\n".format(count_parameters(model)))

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = torch.nn.L1Loss()
scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      mode='min',
                                                      patience=20,
                                                      factor=0.75,
                                                      min_lr=1e-4
                                                      )
#Training loader and Validation loader
batch_size = 128
train_loader, valid_loader = get_train_valid_loader(batch_size=batch_size,random_seed=seed,scale=model.module.scale_factor,bicubic_flag=False,num_workers=4)


train_func(model=model,
             criterion=criterion,
             optimizer=optimizer,
             scheduler=scheduler,
             device=device,
             scale=model.module.scale_factor,
             train_loader=train_loader,
             valid_loader=valid_loader,
             save_path=save_path,
             scheduler_flag=True,
             n_epochs=6000,
             )
