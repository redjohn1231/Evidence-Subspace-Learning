

import tensorflow as tf

from config import Config

from models.network import build_resnet50_backbone
from models.ebd_network import EBDNet

from evidence.subspace import EvidenceSubspace

from dataset import (train_ds,
                     test_ds,
                     val_ds)

from train.trainer import (train_epoch,
                           initialize_evidence,
                           update_evidence)


# ============================================================
# 1. Checkpoint
# ============================================================
def save_checkpoint(model,
                    evidence,
                    path):
    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------
    model.save_weights(path + "model.weights.h5")

    # --------------------------------------------------------
    # Save Evidence Basis
    # --------------------------------------------------------
    tf.io.write_file(path + "P.npy",
                     tf.io.serialize_tensor(evidence.P))

# ============================================================
# 2. Validation
# ============================================================
def validate(model,
             dataset,
             loss_fn):

    loss_metric = tf.keras.metrics.Mean()

    acc_metric = tf.keras.metrics.CategoricalAccuracy()

    for x, y in dataset:

        logits = model(x,training=False)
        
        loss = loss_fn(y,logits)
        
        loss_metric.update_state(loss)
        
        acc_metric.update_state(y,logits)

    return (loss_metric.result().numpy(),
            acc_metric.result().numpy())


# ============================================================
# 3. Relative Structural Change 
# ============================================================
def RS_change(A_new, A_ref):

    A_new = tf.math.l2_normalize(A_new,
                                 axis=0)
    
    A_ref = tf.math.l2_normalize(A_ref,
                                 axis=0)
    
    S_new = tf.matmul(A_new, A_new,
                      transpose_b=True)

    S_ref = tf.matmul(A_ref,
                      A_ref,
                      transpose_b=True)

    diff = tf.norm(S_new - S_ref)
    ref_norm = tf.norm(S_ref)
    delta = diff / (ref_norm + 1e-8)
    
    return delta.numpy()


# ============================================================
# 4. Training Configuration
# ============================================================

WARMUP_EPOCHS = 5

MIN_UPDATE_INTERVAL = 4

CHANGE_THRESHOLD = 0.05

VAL_LOSS_TOL = 1e-3


# ============================================================
# 5. Build Model
# ============================================================
backbone = build_resnet50_backbone()

evidence = EvidenceSubspace(
    Config.FEATURE_DIM,
    Config.EVIDENCE_DIM,
    Config.EMA_BETA)

model = EBDNet(backbone,
               evidence,
               Config.NUM_CLASSES)


# ============================================================
# 6. Initialize Forensic Attributes
# ============================================================

print(
    "\n====================================================")

print(
    "Initialize Forensic Attributes")

print(
    "====================================================")


A_initial = initialize_evidence(model,
                                train_ds)


# ------------------------------------------------------------
# Initial Evidence representation
# ------------------------------------------------------------
evidence.P.assign(A_initial)

# ------------------------------------------------------------
# Initial Evidence Structure Matrix
# ------------------------------------------------------------
Q_initial = tf.matmul(A_initial,
                      A_initial,
                      transpose_b=True)

evidence.Q.assign(Q_initial)

backbone = build_resnet50_backbone(w_name=None)
model = EBDNet(backbone,
               evidence,
               Config.NUM_CLASSES)

# ------------------------------------------------------------
# Reference 
# ------------------------------------------------------------
A_reference = tf.identity(A_initial)


# ============================================================
# 7. Optimizer and Loss
# ============================================================
optimizer = tf.keras.optimizers.Adam(Config.LR)

loss_fn = (tf.keras.losses.CategoricalCrossentropy(from_logits=True))

# ============================================================
# 8. Training State
# ============================================================

prev_val_loss = None
last_update_epoch = 0


# ============================================================
# 9. Training
# ============================================================
for epoch in range(
        Config.EPOCHS):


    print("\n"
          "====================================================")

    print("Epoch:",
          epoch)

    print("====================================================")

    train_epoch(model,
                train_ds,
                optimizer,
                loss_fn)

    val_loss, val_acc = validate(model,
                                 val_ds,
                                 loss_fn)

    print("\nValidation Loss: {:.6f} | "
          "Validation Acc: {:.4f}".format(val_loss,
                                          val_acc)
        )


    converged = False
    
    if prev_val_loss is not None:
        delta_val = abs(val_loss - prev_val_loss)
        
        print("Validation Loss Change: {:.8f}".format(delta_val))


        if (delta_val < VAL_LOSS_TOL):
            
            converged = True

    prev_val_loss = val_loss


    if epoch < WARMUP_EPOCHS:


        print("\n>>> Warm-up Stage")

        print(">>> Evidence Basis is fixed.")

    else:

        print("\n>>> Evidence Basis Learning")

        A_current = initialize_evidence(model,
                                        train_ds)
        
        delta_A = RS_change(A_current,
                            A_reference)

        print("Relative Anchor Structural Change: "
              "{:.8f}".format(delta_A))


        # ----------------------------------------------------
        # Evidence update conditions
        # ----------------------------------------------------
        enough_change = (delta_A > CHANGE_THRESHOLD)


        enough_interval = (epoch -last_update_epoch
                           >=
                           MIN_UPDATE_INTERVAL)

        print("Change condition:",
              enough_change)

        print("Minimum interval condition:",
              enough_interval)

        # ====================================================
        # Trigger Evidence Update
        # ====================================================

        if (enough_change and enough_interval):

            print("\n"
                  "================================================")
            
            print(">>> Evidence Update Triggered")

            print("================================================")

            update_evidence(model,
                            train_ds,
                            evidence,
                            P_new=A_current)


            print(">>> Evidence Basis Updated.")

            A_reference = tf.identity(A_current)

            last_update_epoch = epoch


        # ====================================================
        # Keep Current Evidence Basis
        # ====================================================
        else:
            print("\n>>> Evidence Update Not Triggered.")
            print(">>> Keep Current Evidence Basis.")

    if converged:
        
        print("\n"
              "====================================================")

        print("Training Converged")

        print("====================================================")


        print("Validation Loss Change < {:.1e}".format(VAL_LOSS_TOL))

        checkpoint_path = ("./checkpoints/")

        save_checkpoint(model,
                        evidence,
                        checkpoint_path)

        print("Converged model saved.")

        break


# ============================================================
# 10. Save Final Model
# ============================================================

print("\n====================================================")

print("Save Final Model")

print("====================================================")


checkpoint_path = ("./checkpoints/")

save_checkpoint(model,
                evidence,
                checkpoint_path)

print("Final model saved.")