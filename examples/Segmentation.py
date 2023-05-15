criterion = FocalLoss2d()
scaler = torch.cuda.amp.GradScaler()
from tqdm import tqdm
miou = []
epoch_losses = []
test_losses = []
traintime = []
testtime = []
counter = 0
epoch = 0
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
while counter < 20:  # loop over the dataset multiple times
    t0 = time.time()
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
            segment=encode_segmap(labels)

            with torch.cuda.amp.autocast():
                loss = criterion(outputs, segment.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_accuracy += acc / len(trainloader)
            epoch_loss += loss / len(trainloader)
        
            
            running_loss += loss.item()
            

            tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )

    epoch_losses.append(epoch_loss)
    test_loss = 0
    total = 0

    mIoU = 0
    model.eval()
    with torch.no_grad():
        t1 = time.time()
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # labels = labels.squeeze(1).long()
            outputs = model(images)
            segment=encode_segmap(labels)
#         
            with torch.cuda.amp.autocast():
              test_loss += criterion(outputs, segment.long())
              
            mIoU += iou_pytorch(outputs, segment).mean()
          
    
    mIoU = mIoU/len(testloader)
    test = test_loss/len(testloader)
    test_losses.append(test)
    mIoU = mIoU.cpu().detach()
    miou.append(mIoU)
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    counter += 1
    epoch += 1

    print(f"Epoch : {epoch+1} - MIOU: {mIoU:.4f} -Test Time: {time.time() - t1:.0f} \n")
    if mIoU >= max(miou):
        PATH = 'model.pth'
        torch.save(model.state_dict(), PATH)
        print(1)
        counter = 0
    
print('Finished Training')

model.load_state_dict(torch.load(PATH))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for epoch in range(100):  # loop over the dataset multiple times
    t0 = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    running_loss = 0.0

    model.train()

    with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        for data in tepoch:
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            segment=encode_segmap(labels)

            with torch.cuda.amp.autocast():
                loss = criterion(outputs, segment.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            acc = (outputs.argmax(dim=1) == labels).float().mean()
            epoch_accuracy += acc / len(trainloader)
            epoch_loss += loss / len(trainloader)
        
      
            running_loss += loss.item()


            tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )
    epoch_losses.append(epoch_loss)
    test_loss = 0
    total = 0

    mIoU = 0
    model.eval()
    with torch.no_grad():
        t1 = time.time()
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
           
            outputs = model(images)
            segment=encode_segmap(labels)
#         
            with torch.cuda.amp.autocast():
              test_loss += criterion(outputs, segment.long())
              
            mIoU += iou_pytorch(outputs, segment).mean()
          
    
    mIoU = mIoU/len(testloader)
    test = test_loss/len(testloader)
    test_losses.append(test)
    mIoU = mIoU.cpu().detach()
    miou.append(mIoU)

    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)

    print(f"Epoch : {epoch+1} - MIOU: {mIoU:.4f} -Test Time: {time.time() - t1:.0f} \n")
    if mIoU >= max(miou):
        PATH = 'model.pth'
        torch.save(model.state_dict(), PATH)
        print(1)
    
print('Finished Training')
print(f"Top mIoU : {max(miou):.2f} - Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")

