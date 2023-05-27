import torch

model = YourModel()
optimizer = torch.optim.Adam(model.parameters())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer.to(device)


for epoch in range(num_epochs):
    for batch in data_loader:
        # Move the batch data to the GPU (if available)
        inputs, labels = batch[0].to(device), batch[1].to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = compute_loss(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Detach outputs from the GPU and move to CPU
        outputs = outputs.detach().cpu()

    # Move the model and optimizer to CPU
    model.to("cpu")
    optimizer.to("cpu")
    torch.cuda.empty_cache()  # Clear GPU cache to release memory
