criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()
top1 = []
traintime = []
testtime = []
counter = 0
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False) #Frist optimizer
for epoch in range(100): # loop over the dataset multiple times

    while counter < 20:  #number of epochs of no improvement in performance after which optimizer switches
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
                with torch.cuda.amp.autocast():
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()


                acc = (outputs.argmax(dim=1) == labels).float().mean()
                epoch_accuracy += acc / len(trainloader)
                epoch_loss += loss / len(trainloader)
                tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )

        correct = 0
        total = 0
        correct_1=0
        correct_5=0
        c = 0
        t1 = time.time()
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)


                _, predicted = torch.max(outputs.data, 1)
                res = accuracy(outputs, labels)
                correct_1 += res[0][0].float()
                correct_5 += res[1][0].float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                c += 1
            
        print(f"Epoch : {epoch+1} - Top 1: {correct_1/c:.2f} - Top 5: {correct_5/c:.2f} -Test Time: {time.time() - t1:.0f}\n")

        top1.append(correct_1/c)
        traintime.append(t1 - t0)
        testtime.append(time.time() - t1)
        counter += 1

        if float(correct_1/c) >= float(max(top1)):
            PATH = 'modelname.pth'
            torch.save(model.state_dict(), PATH)
            print(1)
            counter = 0


model.load_state_dict(torch.load(PATH))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) #Second Optimizer


for epoch in range(100):  # loop over the dataset multiple times
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
          with torch.cuda.amp.autocast():
              loss = criterion(outputs, labels)
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()


          acc = (outputs.argmax(dim=1) == labels).float().mean()
          epoch_accuracy += acc / len(trainloader)
          epoch_loss += loss / len(trainloader)
          tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )

    correct = 0
    total = 0
    correct_1=0
    correct_5=0
    c = 0
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)


            _, predicted = torch.max(outputs.data, 1)
            res = accuracy(outputs, labels)
            correct_1 += res[0][0].float()
            correct_5 += res[1][0].float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c += 1
        
    print(f"Epoch : {epoch+1} - Top 1: {correct_1/c:.2f} - Top 5: {correct_5/c:.2f} -Test Time: {time.time() - t1:.0f}\n")

    top1.append(correct_1/c)
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)

    if float(correct_1/c) >= float(max(top1)):
        PATH = 'modelname.pth'
        torch.save(model.state_dict(), PATH)
        print(1)

print('Finished Training')
print(f"Top1 : {max(top1):.2f} - Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")
