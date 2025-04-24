from sklearn.metrics import accuracy_score

# -------------------------
# Evaluation Function (updated for separate modalities)
# -------------------------
def evaluate_model(model, data_loader, device, criterion, torch, moe=False):
    """
    Evaluates the model on the provided data_loader.
    
    Returns:
       avg_loss: float, average loss over the dataset
       acc: float, accuracy over the dataset
       best_val_loss: float, the best (lowest) validation loss observed so far
       best_val_acc: float, the best (highest) validation accuracy observed so far
       
    Note:
       The best validation metrics are maintained as attributes of this function.
       They are updated each time the function is called if the current evaluation 
       shows improvement.
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            # Unpack the batch based on the order:
            #  0: desc_embs, 1: dna_embs, 2: time_embs,
            #  3: desc_input_ids, 4: desc_attention_mask,
            #  5: dna_input_ids, 6: dna_attention_mask,
            #  7: labels
            desc_embs = batch[0].to(device)
            dna_embs = batch[1].to(device)
            time_embs = batch[2].to(device)
            desc_input_ids = batch[3].to(device)
            desc_attention_mask = batch[4].to(device)
            dna_input_ids = batch[5].to(device)
            dna_attention_mask = batch[6].to(device)
            labels = batch[7].to(device)

            # Create tokenized dictionaries for description and DNA branches.
            desc_tokenized = {"input_ids": desc_input_ids, "attention_mask": desc_attention_mask}
            dna_tokenized = {"input_ids": dna_input_ids, "attention_mask": dna_attention_mask}

            # Forward pass
            if moe:
                outputs, aux_loss = model(desc_embs, dna_embs, time_embs, desc_tokenized, dna_tokenized)
            else:
                outputs, aux_loss = model(desc_embs, dna_embs, time_embs, desc_tokenized, dna_tokenized), 0

            loss = criterion(outputs, labels) + aux_loss
            total_loss += loss.item() * desc_embs.size(0)
            total_samples += desc_embs.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)

    # Initialize best values if not already present.
    if not hasattr(evaluate_model, "best_val_acc"):
        evaluate_model.best_val_acc = 0.0
    if not hasattr(evaluate_model, "best_val_loss"):
        evaluate_model.best_val_loss = float('inf')

    # Update best values if improved.
    if acc > evaluate_model.best_val_acc:
        evaluate_model.best_val_acc = acc
    if avg_loss < evaluate_model.best_val_loss:
        evaluate_model.best_val_loss = avg_loss

    return avg_loss, acc, evaluate_model.best_val_loss, evaluate_model.best_val_acc


# -------------------------
# Reusable Training Function (updated for separate modalities)
# -------------------------
def train_model(model, train_loader, val_loader, 
                device, criterion, optimizer, scheduler,
                num_epochs, verbose=False, trial=None, 
                optuna=None, torch=None, moe=False, 
                unfreeze_threshold=0.85):
    # Flag to track if some MOE parameters have been unfrozen.
    unfrozen = False

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_samples = 0
        
        for batch in train_loader:
            # Unpack the batch based on the order:
            #  0: desc_embs, 1: dna_embs, 2: time_embs,
            #  3: desc_input_ids, 4: desc_attention_mask,
            #  5: dna_input_ids, 6: dna_attention_mask,
            #  7: labels
            desc_embs = batch[0].to(device)
            dna_embs = batch[1].to(device)
            time_embs = batch[2].to(device)
            desc_input_ids = batch[3].to(device)
            desc_attention_mask = batch[4].to(device)
            dna_input_ids = batch[5].to(device)
            dna_attention_mask = batch[6].to(device)
            # print(batch)
            labels = batch[7].to(device)
            
            # Create tokenized dictionaries for the model.
            desc_tokenized = {"input_ids": desc_input_ids, "attention_mask": desc_attention_mask}
            dna_tokenized = {"input_ids": dna_input_ids, "attention_mask": dna_attention_mask}

            optimizer.zero_grad()
            if moe:
                outputs, aux_loss = model(desc_embs, dna_embs, time_embs, desc_tokenized, dna_tokenized)
            else:
                outputs, aux_loss = model(desc_embs, dna_embs, time_embs, desc_tokenized, dna_tokenized), 0
            loss = criterion(outputs, labels) + aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1000)
            optimizer.step()
            total_train_loss += loss.item() * desc_embs.size(0)
            total_samples += desc_embs.size(0)
        
        train_loss = total_train_loss / total_samples
        val_loss, val_acc, best_loss, best_acc = evaluate_model(model, val_loader, device, criterion, torch=torch, moe=moe)
        scheduler.step(val_loss)
        
        # Optionally unfreeze router parameters if validation accuracy is high enough.
        if moe and (not unfrozen) and (best_acc > unfreeze_threshold):
            model.dna_moe.w_gate.requires_grad = True
            model.dna_moe.w_noise.requires_grad = True
            model.desc_moe.w_gate.requires_grad = True
            model.desc_moe.w_noise.requires_grad = True
            unfrozen = True
            print("Router params unfrozen!")
        
        if trial:
            trial.report(best_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} "
                  f"- Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} "
                  f"- LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    final_val_loss, final_val_acc, best_loss, best_acc = evaluate_model(model, val_loader, device, criterion, torch=torch, moe=moe)
    return best_acc, best_loss
