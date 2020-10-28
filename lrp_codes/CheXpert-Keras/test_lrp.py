from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import classification_report

from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
import csv

import numpy as np
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score
from utility import get_sample_counts

import innvestigate
import innvestigate.utils

def transparent_cmap(cmap, N=255):
  "Copy colormap and set alpha values"

  mycmap = cmap
  mycmap._init()
  mycmap._lut[:,-1] = np.linspace(0, 0.4, N+4)
  return mycmap

def plot_LRP(heatmap, x1):
  w, h = heatmap.shape
  y, x = np.mgrid[0:h, 0:w]   
  mycmap = transparent_cmap(plt.cm.Reds)
  fig, ax = plt.subplots(1, 1)
  ax.imshow(x1, cmap='gray')
  cb = ax.contourf(x, y, heatmap, 2, cmap=mycmap)
  plt.colorbar(cb)
  plt.savefig("testtt.png")

def main():
    # parser config
    config_file = "./sample_config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")

    # train config
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    test_steps = cp["TEST"].get("test_steps")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, "best_{output_weights_name}")

    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, "valid", class_names)

    # compute steps
    if test_steps == "auto":
        test_steps = int(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError("""
                test_steps: {test_steps} is invalid,
                please use 'auto' or integer.
                """)
    print("** test_steps: {test_steps} **")

    print("** load model **")
    if use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path, input_shape=(image_dimension, image_dimension, 3))

    print("** load test generator **")
    test_sequence = AugmentedImageSequence(
        dataset_csv_file=os.path.join(output_dir, "valid.csv"),
        class_names=class_names,
        source_image_dir=image_source_dir,
        batch_size=batch_size,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=test_steps,
        shuffle_on_epoch_end=False,
    )

    print("** make prediction **")
    y_hat = model.predict_generator(test_sequence, verbose=1)
    y = test_sequence.get_y_true()

    test_log_path = os.path.join(output_dir, "test.log")
    print("** write log to {test_log_path} **")
    aurocs = []
    avg_precisons = []
    with open(test_log_path, "w") as f:
        for i in range(len(class_names)):
            try:
                auroc_score = roc_auc_score(y[:, i], y_hat[:, i])
                avg_precision_score = average_precision_score(y[:, i], y_hat[:, i])
                
                aurocs.append(auroc_score)
                avg_precisons.append(avg_precision_score)
            except ValueError:
                auroc_score = 0
                avg_precision_score_score = 0
            f.write("{class_names[i]}: {score}\n")
            print(str(class_names[i])+ ": AUROC: "+ str(auroc_score) + " : AUPRC: "+str(avg_precision_score))
        mean_auroc = np.mean(aurocs)
        f.write("-------------------------\n")
        f.write("mean auroc: {mean_auroc}\n")
        print("mean auroc: "+str(mean_auroc))

    multi_label_y = []

    count = 0
    np_names = np.array(class_names)
    acc = 0
    for i in range(y.shape[0]):
        if np.argmax(y[i]) == np.argmax(y_hat[i]):
            acc += 1

        ind = y_hat[i].argsort()[-4:][::-1]

        class_name = np_names[ind]


        print(str(i+1)+' '+str(class_name))

    print(acc/y.shape[0])


    '''        
    # Strip softmax layer
    #model = innvestigate.utils.model_wo_softmax(model)
    #print("removed softmax")
    # Create analyzer
    analyzer = innvestigate.create_analyzer("lrp.alpha_2_beta_1", model)

    pathFileValid = '/home/dsv/maul8511/deep_learning_project/data/CheXpert-v1.0-small/valid.csv'


    with open(pathFileValid, "r") as f:
        csvReader = csv.reader(f)
        next(csvReader, None)

        for line in csvReader:

            image_name= '/home/dsv/maul8511/deep_learning_project/data/'+line[0]
            image = Image.open(image_name)
            image_array = np.asarray(image.convert("RGB"))
            image_array = image_array / 255.
            image_array = resize(image_array, (image_dimension, image_dimension))
            image_array = image_array[None, :, :, :]
            

            # Apply analyzer w.r.t. maximum activated output-neuron
            a = analyzer.analyze(image_array)

            

            # Aggregate along color channels and normalize to [-1, 1]
            a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
            a /= np.max(np.abs(a))

            # Plot

            #plot_LRP(a[0], image_array[0])
            
            plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
            plt.savefig('results/lrp_alpha_beta/'+line[0].replace("/","_")+'_lrp_alpha_beta.png')

    '''

if __name__ == "__main__":
    main()
