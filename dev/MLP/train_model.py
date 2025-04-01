from sklearn.metrics import accuracy_score
import torch
# -------------------------
# Evaluation Function (updated for separate modalities)
# -------------------------
def evaluate_model(model, data_loader, device, criterion):
    """
    Evaluates the model on the provided data_loader.
    
    Returns: tuple (average loss, accuracy)
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = [x.to(device) for x in batch[:-1]]
            labels = batch[-1].to(device)
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs[0].size(0)
            total_samples += inputs[0].size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

# -------------------------
# Reusable Training Function (updated for separate modalities)
# -------------------------
def train_model(model, train_loader, val_loader, 
                device, criterion, optimizer, scheduler,
                num_epochs, verbose=False, trial=None, optuna=None):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_samples = 0
        
        for batch in train_loader:
            inputs = [x.to(device) for x in batch[:-1]]
            labels = batch[-1].to(device)
            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * inputs[0].size(0)
            total_samples += inputs[0].size(0)
        
        train_loss = total_train_loss / total_samples
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        if trial:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} "
                  f"- Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} "
                  f"- LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    final_val_loss, final_val_acc = evaluate_model(model, val_loader, device, criterion)
    return final_val_acc, final_val_loss