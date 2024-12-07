### Welcome to my IKT450 project repository!

- **dataset_handler.ipynb** : Contains logic to make the dataset. Takes 1000 generated images per class, 1000 real images per class, and 100 testing images per class and neatly places them in the folder prepared_data/.


- **diffusion_model.ipynb** : Contains the logic to generate the synthetic images. It generates 1000 images per class in batches of 3.


- **CNN_models.ipynb** : Contains the logic to prepare the dataset, train the CNN models on the dataset, and do performance tests with the CNN models.


- **models/** : Contains the three different CNN models, each in their own python file.


- **prepared_data/** : Contains the dataset used to train the CNNs. The generated images are in the gen/ subfolder, the real training images from Places365 are in the train/ subfolder, and the real testing images from Places365 are in the test/ subfolder.