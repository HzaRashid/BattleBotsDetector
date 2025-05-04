from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, average_precision_score
from hybridv3 import MOEAttention
import torch

class Trainer:
    def __init__(self,
                 epochs=20,
                 batch_size=64,
                 optimizer=torch.optim.Adam,
                 lr=1e-4,
                 weight_decay=1e-6):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.lr = lr

        self.model = MOEAttention(num_classes=2,
                                  expert_hidden_dim=128,
                                  top_k=1)
        
        self.init_opt(optimizer)
        self.device = 'cpu'
        self.model.to(self.device)

        self.loss_obj = torch.nn.CrossEntropyLoss()

        self.model_is_moe = True


    def init_opt(self, optimizer):
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.epochs//2], gamma=0.8)


    def train(self,train_loader, val_loader, 
              verbose=False, unfreeze_threshold=1.0):
        # Flag to track if some moe parameters have been unfrozen
        unfrozen = False

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0.0
            total_samples = 0
            
            for batch in train_loader:
                inputs = [x.to(self.device) for x in batch[:-1]]
                labels = batch[-1].to(self.device)
                self.optimizer.zero_grad()
                if self.model_is_moe:
                    outputs, aux_loss = self.model(*inputs)
                else:
                    outputs, aux_loss = self.model(*inputs), 0
                loss = self.loss_obj(outputs, labels) + aux_loss
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 1000)
                self.optimizer.step()
                total_train_loss += loss.item() * inputs[0].size(0)
                total_samples += inputs[0].size(0)
            
            
            train_loss = total_train_loss / total_samples
            (val_loss, val_acc, 
             best_loss, best_acc, f1_score) = self.evaluate_model(val_loader)
            # Step the scheduler
            self.lr_scheduler.step()
            
            # some modality-specific MoE's might benefit from intial freezing
            # of the gates or later freezing the experts:
            if self.model_is_moe and (not unfrozen) and (f1_score > unfreeze_threshold):
                self.param_up()
                unfrozen = True
            
            if verbose:
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f} "
                    f"- Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} "
                    f"- F1 score: {f1_score:.4f}")
                
        final_val_loss, final_val_acc, best_loss, best_acc, f1_score = self.evaluate_model(val_loader)
        return best_acc, best_loss


    def param_up(self, lr_only=True):
        self.optimizer.param_groups[0]['lr']=1e-5
        print('lr updated')
        if lr_only: return

        self.model.dna_moe.w_gate.requires_grad = True
        self.model.dna_moe.w_noise.requires_grad = True
        self.model.desc_tweet_moe.w_gate.requires_grad = True
        self.model.desc_tweet_moe.w_noise.requires_grad = True

        # 1) un‐freeze the gating parameters
        for moe_module in [
            # model.desc_tweet_moe, 
                           self.model.dna_moe
                           ]:
            moe_module.w_gate.requires_grad  = True
            moe_module.w_noise.requires_grad = True

        
        # 2) freeze all of the experts
        for moe_module in [self.model.desc_tweet_moe,
                        #    self.model.dna_moe
                           ]:
            for expert in moe_module.experts:
                for p in expert.parameters():
                    p.requires_grad = False

        print("Params updated")


    # -------------------------
    # Evaluation Function (updated for separate modalities)
    # -------------------------
    @torch.no_grad()
    def evaluate_model(self, data_loader, verbose=False):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        total_samples = 0


        for batch in data_loader:
            inputs = [x.to(self.device) for x in batch[:-1]]
            labels = batch[-1].to(self.device)
            if self.model_is_moe:
                outputs, aux_loss = self.model(*inputs)
            else:
                outputs, aux_loss = self.model(*inputs), 0
            loss = self.loss_obj(outputs, labels) + aux_loss
            total_loss += loss.item() * inputs[0].size(0)
            total_samples += inputs[0].size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / total_samples
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        # Initialize best values as attributes if they don't exist
        if not hasattr(self, "best_val_acc"):
            self.best_val_acc = 0.0
        if not hasattr(self, "best_val_loss"):
            self.best_val_loss = float('inf')
        if not hasattr(self, "best_val_f1"):
            self.best_val_f1 = 0.0

        # Update best values if current metrics are improved
        if acc > self.best_val_acc:
            self.best_val_acc = acc
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1


        if verbose:
            print("Test Accuracy: {:.4f}".format(accuracy_score(all_labels, all_preds)))
            print("Classification Report:\n", classification_report(all_labels, all_preds))
            print("Test ROC AUC: {:.4f}".format(roc_auc_score(all_labels, all_preds)))
            print("Test AUPR: {:.4f}".format(average_precision_score(all_labels, all_preds)))

        return avg_loss, acc, self.best_val_loss, self.best_val_acc, f1


    def get_model(self):
        return self.model