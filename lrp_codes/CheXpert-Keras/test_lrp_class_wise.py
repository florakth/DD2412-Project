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
    

    
    np_names = np.array(class_names)
    heatmap_images = [57, 136, 173, 206, 50, 81, 204, 95, 153, 160, 180, 195, 234, 6, 39, 108, 89, 141, 175]
    

            
    # Strip softmax layer
    #model = innvestigate.utils.model_wo_softmax(model)
    #print("removed softmax")
    # Create analyzer
    analyzer = innvestigate.create_analyzer("lrp.z", model, neuron_selection_mode="index")
    
    lrp_method="lrp_z"
    
    files = ['CheXpert-v1.0-small/valid/patient64582/study1/view1_frontal.jpg',
		'CheXpert-v1.0-small/valid/patient64643/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64679/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64711/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64577/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64599/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64709/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64609/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64660/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64667/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64686/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64701/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64739/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64544/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64568/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64618/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64605/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64648/study1/view1_frontal.jpg',
                'CheXpert-v1.0-small/valid/patient64681/study1/view1_frontal.jpg']
    
    file_id_count = -1
    
    for i in heatmap_images:
        j = i-2
        
        file_name = files[file_id_count]
        file_id_count+=1
        
        image_name= '/home/dsv/maul8511/deep_learning_project/data/'+file_name
        image = Image.open(image_name)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, (image_dimension, image_dimension))
        image_array = image_array[None, :, :, :]
        
        
        
        ind = y_hat[j].argsort()[-4:][::-1]
        
        for output_neuron in ind:
            
            class_name = np_names[output_neuron]
            
            # Apply analyzer w.r.t. maximum activated output-neuron
            a = analyzer.analyze(image_array, neuron_selection=output_neuron)

            

            # Aggregate along color channels and normalize to [-1, 1]
            a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
            a /= np.max(np.abs(a))

            # Plot

            #plot_LRP(a[0], image_array[0])
            
            plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
            plt.savefig('selected_results/'+lrp_method+'/'+file_name.replace("/","_")+'_'+lrp_method+'_'+class_name+'.png')

    

if __name__ == "__main__":
    main()

