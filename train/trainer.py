import tensorflow as tf

from .updater import EMA_update
from evidence.builder import build_basis

def train_epoch(model,dataset,
                optimizer,
                loss_fn):

    steps = tf.data.experimental.cardinality(dataset).numpy()
    
    progbar = tf.keras.utils.Progbar(steps)

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for step,(x,y) in enumerate(dataset):
        with tf.GradientTape() as tape:

            pred = model(x)

            loss = loss_fn(y,
                           pred)

        grad = tape.gradient(loss,
                             model.trainable_variables)

        optimizer.apply_gradients(zip(grad,
                                      model.trainable_variables))

        total_loss += loss.numpy()

        pred_label = tf.argmax(pred,
                               axis=1,
                               output_type=tf.int32)
        
        y = tf.argmax(y,
                      axis=1,
                      output_type=tf.int32)
        
        correct = tf.reduce_sum(tf.cast(pred_label == y,
                                        tf.float32))

        batch_size = tf.shape(y)[0]

        total_correct += correct.numpy()

        total_samples += batch_size.numpy()

        acc = total_correct / total_samples


        progbar.update(step+1,
                       values=[("loss", loss.numpy()),
                               ("acc", acc)])

    epoch_loss = total_loss / steps
    epoch_acc = total_correct / total_samples

    return epoch_loss, epoch_acc




def update_evidence(model,
                    dataset,
                    evidence,
                    P_new):

    feats = []
    labels = []


    total_steps = tf.data.experimental.cardinality(dataset).numpy()

    print("\nUpdating Evidence Subspace...")

    for step, (x, y) in enumerate(dataset):

        f = model.backbone(x,
                           training=False)

        if len(y.shape) > 1:
            y = tf.argmax(y,
                          axis=1,
                          output_type=tf.int32)
        else:
            y = tf.cast(y,
                        tf.int32)

        feats.append(f)
        labels.append(y)

   

        if ((step + 1)
            % max(1, total_steps // 10)
            == 0):

            progress = ((step + 1)
                        / total_steps * 100)

            print(f"Evidence Feature Extraction: "
                  f"{step + 1}/{total_steps} "
                  f"({progress:.1f}%)")

    print("Feature extraction finished.")


    feats = tf.concat(feats,
                      axis=0)

    labels = tf.concat(labels,
                       axis=0)

    total_samples = tf.shape(feats)[0]

    print("Total Evidence Samples:",
          total_samples.numpy())

    print("Feature shape:",
          feats.shape)

    print("Label shape:",
          labels.shape)



    print("\nBuilding Evidence Basis")

    P_new = build_basis(feats,
                        labels)

    print("New Evidence Basis shape:",
          P_new.shape)



    print("\nUpdating Evidence Subspace")

    EMA_update(evidence,
               P_new)

    print("Evidence Subspace update finished.\n")
    
    
def initialize_evidence(model,
                        dataset):

    feats = []
    labels = []
    
    total_steps = tf.data.experimental.cardinality(dataset).numpy()

    print("\nInitializing Evidence Basis...")

    for step, (x, y) in enumerate(dataset):
        
        f = model.backbone(x,
                           training=False)

        if len(y.shape) > 1:
            y = tf.argmax(y,
                          axis=1,
                          output_type=tf.int32)
        else:
            y = tf.cast(y,
                        tf.int32)

        feats.append(f)
        labels.append(y)


        if ((step + 1)
            % max(1, total_steps // 10)
            == 0):

            progress = ((step + 1)
                        / total_steps* 100)

            print(f"Evidence Feature Extraction: "
                  f"{step + 1}/{total_steps} "
                  f"({progress:.1f}%)")

    print("Feature extraction finished.")



    feats = tf.concat(feats,
                      axis=0)

    labels = tf.concat(labels,
                       axis=0)

    print("Total Evidence Samples:",
          tf.shape(feats)[0].numpy())

    print("Feature shape:",
          feats.shape)

    print("Label shape:",
          labels.shape)

    print("\nBuilding Initial Evidence Basis ")

    P_new = build_basis(feats,
                        labels)
    
    print("Initial Evidence Basis shape:",
          P_new.shape)

    print("Evidence Basis initialization finished.\n")

    return P_new
