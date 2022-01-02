import json
import numpy as np
import torch
from torch import nn
from IPython.display import clear_output
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

class Tracker(object):

  def __init__(self, total: bool = False):

    self.is_total = total
    
    self.batch_loss = []
    self.batch_acc = []

    self.train_iter = []
    self.train_loss = []
    self.train_acc = []

    self.val_iter = [] 
    self.val_loss = []
    self.val_acc = []

  def train_update(self, epoch):
    
    self.train_loss.append(np.mean(self.batch_loss))
    self.train_iter.append(epoch + 1)
    if not self.is_total:
      self.train_acc.append(np.mean(self.batch_acc))

    # Reset batch lists
    self.batch_loss = []
    self.batch_acc = []

  def val_update(self, epoch):

    self.val_loss.append(np.mean(self.batch_loss))
    self.val_iter.append(epoch + 1)
    if not self.is_total:
      self.val_acc.append(np.mean(self.batch_acc))
    
    # Reset batch lists
    self.batch_loss = []
    self.batch_acc = []
    
  
  def toJSON(self):
    return json.dumps(self, default=lambda o: o.__dict__)

  def fromJSON(self, load_dict):
      for key in load_dict:
          setattr(self, key, load_dict[key])

def plot_tracker(tracker: Tracker, num_epoch):
    plt.figure(figsize=(14,8))
    epoch_ticks = range(0,num_epoch + 1, 5)

    # loss
    plt.subplot(1,2,1)
    plt.plot(tracker.train_iter, tracker.train_loss, label='Training loss')
    plt.plot(tracker.val_iter, tracker.val_loss, label='Validation loss')
    plt.title("Loss")
    plt.ylabel("Loss"), plt.xlabel("Epoch")
    plt.xticks(epoch_ticks)
    plt.legend()
    plt.grid()

    # acc
    plt.subplot(1,2,2)
    plt.plot(tracker.train_iter, tracker.train_acc, label='Training accuracy')
    plt.plot(tracker.val_iter, tracker.val_acc, label='Validation accuracy')
    plt.title("F1 - Score")
    plt.ylabel("F1 - Score"), plt.xlabel("Epoch")
    plt.xticks(epoch_ticks)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    clear_output(wait=True)



#  =================== MTL HELPERS ================== #

class MTLTracker(object):

    def __init__(self, total: bool = False):
    
        self.is_total = total
        
        self.batch_loss = []
        self.batch_acc = []

        self.train_iter = []
        self.train_loss = []
        self.train_acc = []

        self.val_iter = [] 
        self.val_loss = []
        self.val_acc = []
    
    def train_update(self, epoch):
    
        self.train_loss.append(np.mean(self.batch_loss))
        self.train_iter.append(epoch + 1)
        if not self.is_total:
            self.train_acc.append(np.mean(self.batch_acc))

        # Reset batch lists
        self.batch_loss = []
        self.batch_acc = []

    def val_update(self, epoch):
    
        self.val_loss.append(np.mean(self.batch_loss))
        self.val_iter.append(epoch + 1)
        if not self.is_total:
            self.val_acc.append(np.mean(self.batch_acc))
    
        # Reset batch lists
        self.batch_loss = []
        self.batch_acc = []
    
    
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def fromJSON(self, load_dict):
      for key in load_dict:
          setattr(self, key, load_dict[key])



def plot_MTL_progress(labels: MTLTracker, features: MTLTracker, total: MTLTracker, num_epoch):
      
    plt.figure(figsize=(14,8))
    epoch_ticks = range(0,num_epoch + 1, 5)
    
    # figure for labels
    # loss
    plt.subplot(3,2,1)
    plt.plot(labels.train_iter, labels.train_loss, label='Training loss')
    plt.plot(labels.val_iter, labels.val_loss, label='Validation loss')
    plt.title("Labels: Loss")
    plt.ylabel("Loss"), plt.xlabel("Epoch")
    plt.xticks(epoch_ticks)
    plt.legend()
    plt.grid()
    
    # acc
    plt.subplot(3,2,2)
    plt.plot(labels.train_iter, labels.train_acc, label='Training accuracy')
    plt.plot(labels.val_iter, labels.val_acc, label='Validation accuracy')
    plt.title("Labels: : F1 - Score")
    plt.ylabel("Accuracy"), plt.xlabel("Epoch")
    plt.xticks(epoch_ticks)
    plt.legend()
    plt.grid()
    
    # figure for features
    # loss
    plt.subplot(3,2,3)
    plt.plot(features.train_iter, features.train_loss, label='Training loss')
    plt.plot(features.val_iter, features.val_loss, label='Validation loss')
    plt.title("Features: Loss")
    plt.ylabel("Loss"), plt.xlabel("Epoch")
    plt.xticks(epoch_ticks)
    plt.legend()
    plt.grid()
    
    # acc
    plt.subplot(3,2,4)
    plt.plot(features.train_iter, features.train_acc, label='Training accuracy')
    plt.plot(features.val_iter, features.val_acc, label='Validation accuracy')
    plt.title("Features: F1 - Score")
    plt.ylabel("Score"), plt.xlabel("Epoch")
    plt.xticks(epoch_ticks)
    plt.legend()
    plt.grid()
    
    # figure for total loss
    plt.subplot(3,1,3)
    plt.plot(total.train_iter, total.train_loss, label='Training loss')
    plt.plot(total.val_iter, total.val_loss, label='Validation loss')
    plt.title("Weighted Combined Total Loss")
    plt.ylabel("Loss"), plt.xlabel("Epoch")
    plt.xticks(epoch_ticks)
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    clear_output(wait=True)

class KFoldResult(object):

  def __init__(self, path, isMTL:bool=True):

    def _load_results(path):
      with open(path, "r") as f:
        return json.load(f)

    run_info = _load_results(path)

    self.k         = run_info["k_run"]
    self.num_epoch = run_info["k_epochs"]
    self.train_idx = run_info["train_idx"]
    self.test_idx  = run_info["test_idx"]

    self.test_labels_targets   = np.array(run_info["test_labels_targets"])
    self.test_features_targets = np.array(run_info["test_features_targets"])

    self.test_labels   = torch.Tensor(run_info["test_labels"])
    self.test_features = torch.Tensor(run_info["test_features"])
    
    try:
        self.corrected_preds = torch.Tensor(run_info["corrected_labels"])
    except:
        pass
    
    # labels
    try:
      self.labels_probs = nn.functional.softmax(self.test_labels, dim = 1) 
      self.labels_preds = torch.argmax(self.labels_probs, dim=1).cpu().detach().numpy()
      self.labels_targets = self.test_labels_targets.argmax(axis=1)
    except:
      #print("Skipping labels!")
      pass


    # features
    threshold = 0.5
    try:
      self.features_probs = torch.sigmoid(self.test_features).cpu().detach().numpy()
      self.features_preds = np.array(self.features_probs >= threshold, dtype=float)
      self.features_targets = self.test_features_targets
    except:
      #print("Skipping features!")
      pass

    def _load_tracker(tracker_as_dict, isMTL=True):
      if isMTL: tracker = MTLTracker()
      else: tracker = Tracker()
      tracker.fromJSON(load_dict=tracker_as_dict)
      return tracker
    
    self.labels_tracker = _load_tracker(json.loads(run_info["labels_tracker"]),isMTL)
    self.features_tracker = _load_tracker(json.loads(run_info["features_tracker"]),isMTL)
    self.total_tracker = _load_tracker(json.loads(run_info["total_tracker"]),isMTL)

  def plot_training(self):
    plot_MTL_progress(self.labels_tracker, self.features_tracker, self.total_tracker, self.num_epoch)

  def get_accuracy(self):
    label_acc = f1_score(self.labels_targets, self.labels_preds, average="weighted")
    features_acc = f1_score(self.features_targets, self.features_preds, average="samples")

    return label_acc, features_acc

  def get_labels_accuracy(self):
    return f1_score(self.labels_targets, self.labels_preds, average="weighted")

  def get_features_accuracy(self):
    return f1_score(self.features_targets, self.features_preds, average="samples")


  