from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from My_Dataset import *
from My_Model import *

root_dir = '../datasets/mnist_png'
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

my_train_dataset = MyDataset(root_dir, train=True, transform=transform)
my_test_dataset = MyDataset(root_dir, train=False, transform=transform)

train_dataloader = DataLoader(my_train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(my_test_dataset, batch_size=64, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

writer = SummaryWriter(log_dir='../logs')

model = My_Model().to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch = 5
total_train_step = 0
for i in range(epoch):
    model.train()
    pre_train_step = 0
    pre_train_loss = 0
    for data in train_dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        pre_train_loss += loss.item()
        pre_train_step += 1
        total_train_step += 1
        if pre_train_step % 100 == 0:
            print(f'Epoch {i+1}, Train Loss: {pre_train_loss / pre_train_step}')
            writer.add_scalar('Train Loss', pre_train_loss/pre_train_step, total_train_step)

    model.eval()
    pre_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            pre_accuracy += outputs.argmax(1).eq(labels).sum().item()
    print(f'Test Accuracy: {pre_accuracy/len(my_test_dataset)}')
    writer.add_scalar('Test accuracy', pre_accuracy/len(my_test_dataset), i)
    torch.save(model, f'../models/my_model{i}.pth')

writer.close()
