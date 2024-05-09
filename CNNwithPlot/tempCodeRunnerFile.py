for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2500 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            losses.append(loss.item())
    if (epoch + 1) % 40 == 0:
        learning_rate /= 10
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(f'Updated learning rate: {learning_rate}')
    if (epoch + 1) % 10 == 0:
        print(f'\nTesting after epoch {epoch + 1}:')
        epoch_acc = test_network(model, test_loader, device)
        train_acc = test_network(model, train_loader, device)
        test_accuracies.append(epoch_acc)
        train_accuracies.append(train_acc)
        print('\n')

print('Finished Training')