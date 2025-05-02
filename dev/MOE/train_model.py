from sklearn.metrics import accuracy_score, f1_score

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
       best_f1: float, the best (highest) validation f1 score so far
       
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
            inputs = [x.to(device) for x in batch[:-1]]
            labels = batch[-1].to(device)
            if moe:
                outputs, aux_loss = model(*inputs)
            else:
                outputs, aux_loss = model(*inputs), 0
            loss = criterion(outputs, labels) + aux_loss
            total_loss += loss.item() * inputs[0].size(0)
            total_samples += inputs[0].size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    # Initialize best values as attributes if they don't exist
    if not hasattr(evaluate_model, "best_val_acc"):
        evaluate_model.best_val_acc = 0.0
    if not hasattr(evaluate_model, "best_val_loss"):
        evaluate_model.best_val_loss = float('inf')
    if not hasattr(evaluate_model, "best_val_f1"):
        evaluate_model.best_val_f1 = 0.0

    # Update best values if current metrics are improved
    if acc > evaluate_model.best_val_acc:
        evaluate_model.best_val_acc = acc
    if avg_loss < evaluate_model.best_val_loss:
        evaluate_model.best_val_loss = avg_loss
    if f1 > evaluate_model.best_val_f1:
         evaluate_model.best_val_f1 = f1

    return avg_loss, acc, evaluate_model.best_val_loss, evaluate_model.best_val_acc, evaluate_model.best_val_f1


# -------------------------
# Reusable Training Function (updated for separate modalities)
# -------------------------
def train_model(model, train_loader, val_loader, 
                device, criterion, optimizer, scheduler,
                num_epochs, verbose=False, trial=None, 
                optuna=None, torch=None, moe=False, 
                unfreeze_threshold=0.50):
    # Flag to track if some moe parameters have been unfrozen
    unfrozen = False

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_samples = 0
        
        for batch in train_loader:
            inputs = [x.to(device) for x in batch[:-1]]
            labels = batch[-1].to(device)
            optimizer.zero_grad()
            if moe:
                outputs, aux_loss = model(*inputs)
            else:
                outputs, aux_loss = model(*inputs), 0
            loss = criterion(outputs, labels) + aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1000)
            optimizer.step()
            total_train_loss += loss.item() * inputs[0].size(0)
            total_samples += inputs[0].size(0)
        
        
        train_loss = total_train_loss / total_samples
        val_loss, val_acc, best_loss, best_acc, best_f1 = evaluate_model(model, val_loader, device, criterion, torch=torch, moe=moe)
        # Step the scheduler
        scheduler.step()
        
        # Check if we should unfreeze the router parameters.
        # if moe and (not unfrozen) and (best_f1 > unfreeze_threshold):
        #     optimizer.param_groups[0]['lr']=1e-5
        #     # model.dna_moe.w_gate.requires_grad = True
        #     # model.dna_moe.w_noise.requires_grad = True
        #     # model.desc_tweet_moe.w_gate.requires_grad = True
        #     # model.desc_tweet_moe.w_noise.requires_grad = True

        #     # 1) un‐freeze the gating parameters
        #     for moe_module in [
        #         # model.desc_tweet_moe, 
        #                        model.dna_moe
        #                        ]:
        #         moe_module.w_gate.requires_grad  = True
        #         moe_module.w_noise.requires_grad = True

            
        #     # 2) freeze all of the experts
        #     for moe_module in [model.desc_tweet_moe,
        #                     #    model.dna_moe
        #                        ]:
        #         for expert in moe_module.experts:
        #             for p in expert.parameters():
        #                 p.requires_grad = False

            # unfrozen = True
            # print("Router params unfrozen!")
        
        if trial:
            trial.report(best_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} "
                  f"- Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} "
                  f"- LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    final_val_loss, final_val_acc, best_loss, best_acc, best_f1 = evaluate_model(model, val_loader, device, criterion, torch=torch, moe=moe)
    return best_acc, best_loss