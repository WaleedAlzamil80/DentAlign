import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(model, optimizer, scheduler, train_loader, val_loader, device, args):
    train_loss = 0
    print("Start learning: ")
    for epoch in range(args.num_epochs):
        model.train()

        for images, masked_images in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            images, masked_images = images.to(device), masked_images.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            reconstructed_images, mu, logvar = model(masked_images)
            
            # Compute loss
            loss = loss_function(reconstructed_images, images, mu, logvar)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()
        
        # Apply learning rate scheduler if it's used
        if args.use_scheduler:
            scheduler.step()

        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {train_loss / len(train_loader.dataset)}")
        
        # Validation step at the end of each epoch (optional)
        validate(model, val_loader, device, epoch, args)


def validate(model, val_loader, device, epoch, args):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masked_images in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            images, masked_images = images.to(device), masked_images.to(device)

            reconstructed_images, mu, logvar = model(masked_images)

            loss = loss_function(reconstructed_images, images, mu, logvar)
            val_loss += loss.item()
    
    val_loss /= len(val_loader.dataset)
    print(f"Validation Loss: {val_loss}")


def loss_function(reconstructed_x, x, mu, logvar):
    # Reconstruction loss (Binary Cross Entropy)
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')

    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_div