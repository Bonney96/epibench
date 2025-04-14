import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast # Import for mixed precision
import gc # For garbage collection if needed
from typing import Optional, Dict, Any, Callable # For type hinting + Callable
import optuna

# Use the root logger configured by LoggerManager
logger = logging.getLogger(__name__) # Get logger for this module

class Trainer:
    """
    Handles the training and validation loops for a PyTorch model.
    Includes TensorBoard logging, checkpoint saving, and optional mixed-precision training.
    Configurable via a dictionary.
    """
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module,
                 config: Dict[str, Any], 
                 train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                 pruning_callback: Optional[Callable[[int, float], None]] = None # Add pruning callback
                 ):
        """
        Initializes the Trainer, configuring settings from the config dictionary.

        Args:
            model: The model to train.
            optimizer: The optimizer to use.
            criterion: The loss function.
            config: Configuration dictionary containing settings like:
                - epochs (int, default: 10)
                - use_mixed_precision (bool, default: False)
                - use_gradient_checkpointing (bool, default: False) # Model needs to implement this
                - checkpoint_dir (str, default: 'checkpoints')
                - log_dir (str, default: 'runs/epibench_experiment_<timestamp>')
                - save_best_only (bool, default: True) # If true, only save best model
                - save_every_n_epochs (int, optional): Save checkpoint every N epochs regardless of performance.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            device: The device to run training on (e.g., 'cuda' or 'cpu').
            pruning_callback: Optional callback for Optuna pruning (epoch, val_loss) -> None.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self._pruning_callback = pruning_callback # Store the callback

        # --- Configuration derived attributes ---
        self.epochs = self.config.get('epochs', 10)
        if not isinstance(self.epochs, int) or self.epochs <= 0:
            logger.warning(f"Invalid 'epochs' value ({self.epochs}). Using default 10.")
            self.epochs = 10
            
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        default_log_dir = f"runs/epibench_experiment_{timestamp}"
        default_checkpoint_dir = f"checkpoints/epibench_experiment_{timestamp}"
        
        self.log_dir = self.config.get('log_dir', default_log_dir)
        self.checkpoint_dir = self.config.get('checkpoint_dir', default_checkpoint_dir)
        self.save_best_only = self.config.get('save_best_only', True)
        self.save_every_n_epochs = self.config.get('save_every_n_epochs', None)
        if self.save_every_n_epochs is not None and (not isinstance(self.save_every_n_epochs, int) or self.save_every_n_epochs <= 0):
             logger.warning(f"Invalid 'save_every_n_epochs' ({self.save_every_n_epochs}). Disabling periodic saving.")
             self.save_every_n_epochs = None

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True) # Ensure log dir exists
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # --- Mixed Precision Setup ---
        self.use_mixed_precision = self.config.get('use_mixed_precision', False) and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_mixed_precision)
        if self.use_mixed_precision:
            logger.info("Mixed precision training enabled.")
        else:
            logger.info("Mixed precision training disabled.")
            
        # Note: Gradient checkpointing needs to be implemented within the model's forward pass
        self.use_gradient_checkpointing = self.config.get('use_gradient_checkpointing', False)
        if self.use_gradient_checkpointing:
             logger.info("Gradient Checkpointing configured (ensure model implements it).")

    def _save_checkpoint(self, is_best=False):
        """Saves the current model state as a checkpoint.
        
        Handles saving best model vs periodic saving.
        
        Args:
            is_best (bool): If True, saves as 'best_model.pth'.
        """
        if is_best:
            filename = "best_model.pth"
        elif self.save_every_n_epochs and (self.current_epoch + 1) % self.save_every_n_epochs == 0:
             filename = f"epoch_{self.current_epoch + 1}.pth"
        else:
            # Don't save if not best and periodic saving is not active or not the right epoch
             return
             
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        state = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_mixed_precision else None, 
            'best_val_loss': self.best_val_loss,
            'config': self.config # Save config used for this run
        }
        try:
             torch.save(state, checkpoint_path)
             logger.info(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
             logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}", exc_info=True)

    def _log_gpu_memory(self, stage="EpochEnd"):
        """Logs current GPU memory usage if CUDA is available."""
        if self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated(self.device) / (1024**2) # MB
                reserved = torch.cuda.memory_reserved(self.device) / (1024**2) # MB
                max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**2) # MB
                max_reserved = torch.cuda.max_memory_reserved(self.device) / (1024**2) # MB
                
                log_message = (f"GPU Memory ({stage} - Epoch {self.current_epoch+1}): "
                               f"Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB, "
                               f"MaxAllocated={max_allocated:.2f}MB, MaxReserved={max_reserved:.2f}MB")
                logger.info(log_message)
                
                # Log to TensorBoard
                self.writer.add_scalar(f'Memory/Allocated_{stage}', allocated, self.current_epoch)
                self.writer.add_scalar(f'Memory/Reserved_{stage}', reserved, self.current_epoch)
                # Reset peak memory stats for next epoch tracking if desired
                # torch.cuda.reset_peak_memory_stats(self.device)
            except Exception as e:
                 logger.error(f"Failed to log GPU memory: {e}", exc_info=True)

    def train_one_epoch(self):
        """
        Runs a single training epoch.
        """
        self.model.train()  # Set model to training mode
        total_loss = 0.0
        num_batches = len(self.train_loader)

        logger.info(f"Starting Training Epoch {self.current_epoch+1}/{self.epochs}")

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs} Training", leave=False)

        for batch_idx, (features, targets) in enumerate(pbar):
            features, targets = features.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True) # More memory efficient

            with autocast(enabled=self.use_mixed_precision):
                outputs = self.model(features)
                if outputs is None:
                     logger.error(f"Model returned None output at epoch {self.current_epoch+1}, batch {batch_idx}. Skipping batch.")
                     continue # Skip this batch if model output is None
                try:
                    loss = self.criterion(outputs, targets)
                except Exception as e:
                     logger.error(f"Error computing loss at epoch {self.current_epoch+1}, batch {batch_idx}: {e}", exc_info=True)
                     logger.error(f"Output shape: {outputs.shape}, Target shape: {targets.shape}")
                     continue # Skip this batch

            if loss is None:
                 logger.error(f"Loss computation returned None at epoch {self.current_epoch+1}, batch {batch_idx}. Skipping backward pass.")
                 continue # Skip if loss is None
                 
            # Check for NaN/inf loss before scaling
            if not torch.isfinite(loss):
                 logger.error(f"Non-finite loss detected at epoch {self.current_epoch+1}, batch {batch_idx}: {loss.item()}. Skipping backward pass.")
                 continue # Skip backward/step if loss is NaN/inf
                 
            self.scaler.scale(loss).backward()
            # Optional: Gradient clipping can be added here if needed
            # self.scaler.unscale_(self.optimizer) # Unscale before clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = loss.item()
            total_loss += batch_loss
            
            # Log batch loss to TensorBoard (optional, can be verbose)
            # global_step = self.current_epoch * num_batches + batch_idx
            # self.writer.add_scalar('Loss/train_batch', batch_loss, global_step)
            
            pbar.set_postfix({'batch_loss': f"{batch_loss:.4f}"})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        pbar.close()
        logger.info(f"Finished Training Epoch {self.current_epoch+1}/{self.epochs}. Average Loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.
        Reports intermediate results for pruning if callback is provided.
        """
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0.0
        num_batches = len(self.val_loader)
        if num_batches == 0:
             logger.warning("Validation loader is empty. Skipping validation.")
             # Report a high loss if pruning is active, otherwise return inf
             if self._pruning_callback:
                 try:
                     # Report a very high value to potentially trigger pruning
                     self._pruning_callback(self.current_epoch, float('inf')) 
                 except optuna.TrialPruned: # Catch pruning exception if it happens here
                     raise # Re-raise immediately
                 except Exception as e:
                     logger.error(f"Error in pruning callback during empty validation: {e}", exc_info=True)
             return float('inf') 

        logger.info(f"Starting Validation Epoch {self.current_epoch+1}/{self.epochs}")
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1}/{self.epochs} Validation", leave=False)

        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(pbar):
                features, targets = features.to(self.device), targets.to(self.device)

                with autocast(enabled=self.use_mixed_precision):
                     try:
                        outputs = self.model(features)
                        if outputs is None:
                             logger.error(f"Model returned None output during validation at epoch {self.current_epoch+1}, batch {batch_idx}. Skipping batch.")
                             continue
                        loss = self.criterion(outputs, targets)
                     except Exception as e:
                          logger.error(f"Error during validation forward pass at epoch {self.current_epoch+1}, batch {batch_idx}: {e}", exc_info=True)
                          continue
                          
                if loss is None or not torch.isfinite(loss):
                    logger.error(f"Non-finite or None validation loss detected at epoch {self.current_epoch+1}, batch {batch_idx}. Skipping batch.")
                    continue

                batch_loss = loss.item()
                total_loss += batch_loss
                pbar.set_postfix({'val_loss': f"{batch_loss:.4f}"})

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        pbar.close()
        logger.info(f"Finished Validation Epoch {self.current_epoch+1}/{self.epochs}. Average Loss: {avg_loss:.4f}")
        
        # --- Pruning Callback --- 
        if self._pruning_callback:
            try:
                # Report the average validation loss for this epoch
                self._pruning_callback(self.current_epoch, avg_loss)
                logger.debug(f"Reported val_loss {avg_loss:.4f} for epoch {self.current_epoch} to Optuna trial.")
            except optuna.TrialPruned: # Catch pruning exception
                 logger.info(f"Pruning condition met after epoch {self.current_epoch}. Raising TrialPruned.")
                 raise # Re-raise for HPOptimizer.objective to catch
            except Exception as e:
                 # Log error but don't stop training just because of callback error
                 logger.error(f"Error executing pruning callback: {e}", exc_info=True)
                 
        return avg_loss

    def train(self):
        """
        Runs the full training loop for the specified number of epochs.
        Handles TrialPruned exceptions raised by the validation callback.
        """
        # Now using self.epochs directly which was set in __init__
        logger.info(f"Starting training run for {self.epochs} epochs on {self.device}.")
        logger.info(f"Mixed Precision: {'Enabled' if self.use_mixed_precision else 'Disabled'}")
        logger.info(f"Gradient Checkpointing: {'Enabled (model must support)' if self.use_gradient_checkpointing else 'Disabled'}")
        logger.info(f"Checkpoints will be saved in: {self.checkpoint_dir}")
        logger.info(f"Logs will be saved in: {self.log_dir}")
        logger.info(f"Saving best model only: {self.save_best_only}")
        if self.save_every_n_epochs:
            logger.info(f"Saving checkpoint every {self.save_every_n_epochs} epochs.")

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            try:
                train_loss = self.train_one_epoch()
                val_loss = self.validate() # This might raise TrialPruned

                # Log average epoch losses to TensorBoard
                self.writer.add_scalar('Loss/train_epoch', train_loss, self.current_epoch)
                self.writer.add_scalar('Loss/validation_epoch', val_loss, self.current_epoch)
                
                # Log learning rate (if using a scheduler, get it from there)
                try:
                     current_lr = self.optimizer.param_groups[0]['lr']
                     self.writer.add_scalar('LearningRate', current_lr, self.current_epoch)
                     logger.info(f"Epoch {self.current_epoch+1} Learning Rate: {current_lr:.6f}") # More precision for LR
                except IndexError:
                     logger.warning("Could not log learning rate (optimizer may not have param_groups).")
                     
                # Log GPU memory usage at the end of the epoch
                self._log_gpu_memory(stage="EpochEnd")

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    logger.info(f"New best validation loss: {self.best_val_loss:.4f}. Saving best model...")
                    self._save_checkpoint(is_best=True)
                    
                # Save periodically if configured and not saving only best
                if not self.save_best_only and self.save_every_n_epochs:
                     self._save_checkpoint(is_best=False)
            
            except optuna.TrialPruned: 
                # If validate() raised TrialPruned, catch it here, log, and break the loop
                logger.info(f"Trial pruned at epoch {self.current_epoch}. Stopping training for this trial.")
                # self.writer.close() # Close writer if stopping early
                break # Exit the epoch loop for this trial
            except Exception as e:
                 logger.error(f"Error during epoch {self.current_epoch+1}: {e}", exc_info=True)
                 # Optionally break or continue based on severity
                 break # Example: Stop training on other errors too

        logger.info(f"Training loop finished for this trial after {self.current_epoch + 1} epochs. Best Validation Loss: {self.best_val_loss:.4f}")
        self.writer.close()
        # Note: The best_val_loss is returned by HPOptimizer.objective

    @staticmethod
    def load_model(checkpoint_path, model, device, optimizer=None, scaler=None):
        """
        Loads model, optimizer, and scaler states from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file (.pth).
            model (nn.Module): An instance of the model architecture.
            device (torch.device): The device to load onto.
            optimizer (Optional[optim.Optimizer]): Optimizer instance to load state into.
            scaler (Optional[GradScaler]): GradScaler instance to load state into.

        Returns:
            Tuple[nn.Module, Optional[optim.Optimizer], Optional[GradScaler], dict]: 
                Model, Optimizer (if provided), Scaler (if provided), and the full checkpoint dict.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            KeyError: If required keys are missing from the checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' not in checkpoint:
             raise KeyError(f"'model_state_dict' not found in checkpoint: {checkpoint_path}")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        logger.info(f"Model state loaded from epoch {checkpoint.get('epoch', 'N/A')} in {checkpoint_path}")

        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state loaded.")
            except Exception as e:
                 logger.warning(f"Could not load optimizer state: {e}")
        elif optimizer:
             logger.warning("Optimizer provided but 'optimizer_state_dict' not found in checkpoint.")

        if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info("GradScaler state loaded.")
            except Exception as e:
                 logger.warning(f"Could not load GradScaler state: {e}")
        elif scaler and scaler.is_enabled(): # Only warn if scaler was expected (i.e., mixed precision enabled)
             logger.warning("GradScaler provided but 'scaler_state_dict' not found or was None in checkpoint.")

        model.eval() # Set model to evaluation mode by default after loading
        return model, optimizer, scaler, checkpoint

# Example usage needs adjustment if loading optimizer/scaler state
# ... (rest of example usage) ...

# Example usage (commented out - for illustration)
# if __name__ == '__main__':
#     # 1. Dummy Data and Loaders
#     class DummyDataset(torch.utils.data.Dataset):
#         def __init__(self, num_samples=1000, feature_dim=10, target_dim=1):
#             self.num_samples = num_samples
#             self.features = torch.randn(num_samples, feature_dim)
#             self.targets = torch.randn(num_samples, target_dim) # Example for regression
#             # For classification, use: torch.randint(0, 2, (num_samples,)).float().unsqueeze(1)

#         def __len__(self):
#             return self.num_samples

#         def __getitem__(self, idx):
#             return self.features[idx], self.targets[idx]

#     train_dataset = DummyDataset(num_samples=800)
#     val_dataset = DummyDataset(num_samples=200)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#     # 2. Dummy Model, Optimizer, Criterion
#     model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1)) # Simple MLP
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.MSELoss() # Mean Squared Error for regression
#     # For binary classification use: nn.BCEWithLogitsLoss()

#     # 3. Config and Device
#     config = {'epochs': 5, 'learning_rate': 0.001}
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
#     log_dir = f"runs/dummy_experiment_{timestamp}" # Example log dir
#     checkpoint_dir = f"checkpoints/dummy_experiment_{timestamp}" # Example checkpoint dir

#     # 4. Initialize and Run Trainer
#     trainer = Trainer(model, optimizer, criterion, config, train_loader, val_loader, device, log_dir=log_dir, checkpoint_dir=checkpoint_dir) # Pass dirs
#     trainer.train()
#     print(f"Training complete. Best validation loss: {trainer.best_val_loss:.4f}")
#     print(f"Checkpoints saved in: {checkpoint_dir}")
#     print(f"TensorBoard logs saved to: {log_dir}")
#     # To view: tensorboard --logdir runs

#     # 5. Example Loading the best model
#     # best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
#     # # Create a new model instance (or use the same one)
#     # loaded_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
#     # try:
#     #     loaded_model = Trainer.load_model(best_checkpoint_path, loaded_model, device)
#     #     print("Successfully loaded model from checkpoint.")
#     #     # Now you can use loaded_model for inference
#     #     # Example: loaded_model(torch.randn(1, 10).to(device))
#     # except (FileNotFoundError, KeyError) as e:
#     #      print(f"Error loading model: {e}") 