import torch, time
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from torchmetrics.classification import Accuracy
from lion_pytorch import Lion


def post_train(model, trainloader, testloader, device, PATH, top1, top5, traintime, testtime, num_classes, opt1 = 'adamw', pretrained = False, set_counter = 20):
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  print('Post-training with SGD')

  top1_acc = Accuracy(task="multiclass", num_classes=10).to(device)
  top5_acc = Accuracy(task="multiclass", num_classes=10, top_k=5).to(device)

  criterion = nn.CrossEntropyLoss()

  scaler = torch.cuda.amp.GradScaler()

  epoch = 0
  counter = 0
  while counter < set_counter:   #Counter sets the number of epochs of non improvement before stopping

      t0 = time.time()

      train(model, trainloader, criterion, scaler, optimizer, epoch)

      correct_1=0
      correct_5=0
      c = 0
      model.eval()
      t1 = time.time()
      with torch.no_grad():
          for data in testloader:
              images, labels = data[0].to(device), data[1].to(device)
              outputs = model(images)
              correct_1 += top1_acc(outputs, labels)
              correct_5 += top5_acc(outputs, labels)
              c += 1
          
      print(f"Epoch : {epoch+1} - Top 1: {correct_1*100/c:.2f} - Top 5: {correct_5*100/c:.2f} -  Train Time: {t1 - t0:.2f} - Test Time: {time.time() - t1:.2f}\n")

      top1.append(correct_1*100/c)
      top5.append(correct_5*100/c)
      traintime.append(t1 - t0)
      testtime.append(time.time() - t1)
      counter += 1
      epoch += 1
      if float(correct_1*100/c) >= float(max(top1)):
          torch.save(model.state_dict(), PATH)
          print(1)
          counter = 0


def train(model, trainloader, criterion, scaler, optimizer, epoch):
  epoch_accuracy = 0
  epoch_loss = 0
  running_loss = 0.0
  model.train()
  with tqdm(trainloader, unit="batch") as tepoch:
          tepoch.set_description(f"Epoch {epoch+1}")

          for data in tepoch:
    
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            with torch.cuda.amp.autocast():
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_accuracy += acc / len(trainloader)
            epoch_loss += loss / len(trainloader)
            tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )


def classification(model, trainloader, testloader, device, PATH, top1, top5, traintime, testtime, num_classes, opt1 = 'adamw', pretrained = False, set_counter = 20):

  criterion = nn.CrossEntropyLoss()

  scaler = torch.cuda.amp.GradScaler()

  top1_acc = Accuracy(task="multiclass", num_classes=10).to(device)
  top5_acc = Accuracy(task="multiclass", num_classes=10, top_k=5).to(device)


  counter = 0

  if opt1 == 'lion':
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
    print("Training with Lion")

  else:
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    print("Training with AdamW")

  # load saved model
  if pretrained:
    PATH = 'model.pth'
    model.load_state_dict(torch.load(PATH))

  epoch = 0
  while counter < set_counter:   #Counter sets the number of epochs of non improvement before stopping

      t0 = time.time()

      train(model, trainloader, criterion, scaler, optimizer, epoch)

      correct_1=0
      correct_5=0
      c = 0
      model.eval()

      t1 = time.time()
      with torch.no_grad():
          for data in testloader:
              images, labels = data[0].to(device), data[1].to(device)
              outputs = model(images)
              correct_1 += top1_acc(outputs, labels)
              correct_5 += top5_acc(outputs, labels)
              c += 1
          
      print(f"Epoch : {epoch+1} - Top 1: {correct_1*100/c:.2f} - Top 5: {correct_5*100/c:.2f} -  Train Time: {t1 - t0:.2f} - Test Time: {time.time() - t1:.2f}\n")

      top1.append(correct_1*100/c)
      top5.append(correct_5*100/c)
      traintime.append(t1 - t0)
      testtime.append(time.time() - t1)
      counter += 1
      epoch += 1
      if float(correct_1*100/c) >= float(max(top1)):
          torch.save(model.state_dict(), PATH)
          print(1)
          counter = 0

  print('Finished Training')
  
  model.load_state_dict(torch.load(PATH))
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  print('Training with SGD')

  epoch = 0
  counter = 0
  while counter < set_counter:   #Counter sets the number of epochs of non improvement before stopping

      t0 = time.time()

      train(model, trainloader, criterion, scaler, optimizer, epoch)

      correct_1=0
      correct_5=0
      c = 0
      model.eval()
      t1 = time.time()
      with torch.no_grad():
          for data in testloader:
              images, labels = data[0].to(device), data[1].to(device)
              outputs = model(images)
              correct_1 += top1_acc(outputs, labels)
              correct_5 += top5_acc(outputs, labels)
              c += 1
          
      print(f"Epoch : {epoch+1} - Top 1: {correct_1*100/c:.2f} - Top 5: {correct_5*100/c:.2f} -  Train Time: {t1 - t0:.2f} - Test Time: {time.time() - t1:.2f}\n")

      top1.append(correct_1*100/c)
      top5.append(correct_5*100/c)
      traintime.append(t1 - t0)
      testtime.append(time.time() - t1)
      counter += 1
      epoch += 1
      if float(correct_1*100/c) >= float(max(top1)):
          torch.save(model.state_dict(), PATH)
          print(1)
          counter = 0


      


