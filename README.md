# Object Detection and Segmentation
# A Step-by-step guide to Pytorch Detectron2 with custom dataset
# Environments:
1. Google Colab (strongly recommended, .ipynb file is provided.)
2. Ubuntu (We will use in this tutorial.)

## For Google Colab:
Download a copy of the .ipynb file, run it on Google Colab.

## For Ubuntu:
### Step 1: Dealing with Anaconda (You may skip step 1, Anaconda is NOT compulsory.)
* Install Anaconda, since we will use Anaconda to manage environments.\
[How to install Anaconda](https://docs.anaconda.com/anaconda/install/linux/)
* Open your terminal, create a new Python environment. (We named the environment 'detectron2').\
`~$ conda create -n detectron2`
* Activate the environment we have created.\
`~$ conda activate detectron2`
* If the above steps are successful, you will see something similar to this:\
`(detectron2) lemon@lemon-System-Product-Name:~$`

### Step 2: Install Detectron2 
[How to install Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
#### Method 1: Which I was using (recommended).
1. Change the current working directory to Desktop. You can change to any other directory or remain the same.\
`~$ cd Desktop `
2. Clone the repository.\
` ~$ git clone https://github.com/facebookresearch/detectron2.git`
3. Install detectron2\
` ~$ python3 -m pip install -e detectron2`
4. List the folders in current directory. You will now see a folder called "detectron2".\
`~$ ls`\
![Image of terminal](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/blob/main/resources/step2.png)
#### Method 2:
1. Check gcc version, since version >= 5.4 is needed.\
   `~$ gcc --version`
2. Check torch version and GPU availability.
   ```
   ~$ python3
   >>> import torch, torchvision
   >>> print(torch.__version__, torch.cuda.is_available())
   1.8.1+cu102 True
   >>> exit()
   ```
3. Go to [How to install Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), locate     "**Install Pre-Built Detectron2 (Linux only)**". Then, from the table, choose the correct version. In our case, we select **torch 1.8** and **CUDA 10.2**. (These are what we saw in the above output)
4. Copy the command and run it on your terminal.\
   `~$ python3 python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html`

### Step 3: Test our installation
1. Download a picture for testing purpose.\
   `~$ wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg`
2. Run test_installation.py\
   _Please open the file(test_installation.py) with text editor, comment line25 `cfg.MODEL.DEVICE='cpu' `, if you do have a GPU available._\
` ~$ python3 test_installation.py`
3. You will now see the original and predicted images like below. Press 'space' to terminate the program.
![Original image](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/blob/main/resources/input.jpg)
![Predicted image](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/blob/main/resources/result.jpg)

### Step 4: Prepare our dataset
#### First: Label your images
* We use the tool [labelme](https://github.com/wkentaro/labelme) to label our images.
An example for the image and corresponding annotation can be found in resources/sample [here](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/tree/main/resources/sample).
* There are many tutorials guiding you to use labelme, you may check them out.
#### Second: Register your dataset
* Since we must register our dataset in Detectron2 in order to use it [see here](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html).
There are few methods to do it.
##### Method 1: Convert annotations to COCO Format dataset.(Fast and simple, highly recommended.)
- A conversion tool is provided in my [Github](https://github.com/fookseng/annotation-converter).
- After you have done the conversion, we are able to use the function 'register_coco_instances' to register our datasets easily (in Step 5).
- A sample output file from the conversion can be found in resources/sample/sample.json [sample](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/blob/main/resources/sample/sample.json)
##### Method 2: Write your own function to register the dataset.(Not tested in this tutorial.)
If you are using labelme, a sample code is provided for your reference(register_dataset.py).
Credit to [Jadezzz](https://github.com/Jadezzz)
* #### Conclusion: In Step 4, we label our images using labelme. Then, we convert all the annotations to COCO Format(sample.json). What we are doing:
   1. Collect all your images, put them all together in a folder named 'img'.
   2. Label all the images using labelme, save the annotations in another folder named 'label'
   3. To use the tool(convert labelme annotation to COCO Format) mentioned above, we must have both the images and annotations in the same folder. Hence, we create a new folder named 'coco', then, copy all the images and annotations to the folder 'coco'.
   4. Run the tool to do the conversion (using the folder 'coco').
   5. Now, we have obtained the COCO Format annotation file(sample.json). You may delete the folder 'coco'.
   
   * You might want to split your data into 2 sets(Train Set, Valid Set) or 3 sets(Test Set, Valid Set, Train Set). Then, repeat the above process.

### Step 5: Let's start to train!
First of all, the below is my file structure.
...
...
Open the file 'train.py' with text editor. We are going to modify some settings.
* Compulsory
   1. From line 31-70, you can see a block of code like below image. Read the comment for each variable, then modify them. The comment should be clear enough for you to understand.
   ![modify](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/blob/main/resources/code_block.png)
* Optional
   1. You may go through the code and modify other configurations based on your needs.
      - Data augmentation [documentation](https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html)
      - Custom Trainer
      - config-file
      - model
      - learning rate
      - [more](https://detectron2.readthedocs.io/en/latest/modules/config.html)
      
1. The function 'custom_mapper', apply some transformations to our data. You may edit this function based on your needs.
2. The class 'CocoTrainer' is a custom Trainer derived from DefaultTrainer. Also, you may edit this class based on your needs.[Reference](https://github.com/facebookresearch/detectron2/blob/master/projects/DeepLab/train_net.py)

* #### Conclusion: You can keep everything unchanged, but MUST change all the file's path, unless you follow my file structure exactly the same.

Save the file, and now, let's start our training. Run the file 'train.py'.\
`~$ python3 train.py`

_You will first see 3 images displayed one by one(press 'space' to skip to the next one). This is to confirm we have registered our dataset correctly. After the 3rd image displayed, the training process will start._\
![training](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/blob/main/resources/train.png)

### Step 6: Open Tensorboard to display the performance.
First, open a new terminal, type the following command.\
Make sure you are in the same directory with the folder 'output'.\
`~$ tensorboard --logdir output`
![tensor1](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/blob/main/resources/tensorboard1.png)
![tensor2](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/blob/main/resources/tensorboard2.png)
_You might have to install tensorboard before you can use it. Check online tutorial for the installation of Tensorboard._
 
### Step 7: Evaluate
Evaluation information will be printed on the terminal.\
![evaluation](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/blob/main/resources/evaluate.png)

### Step 8: Inference-Test your model.
* I am going to use the code provided in [Github](https://github.com/facebookresearch/detectron2/tree/master/demo) to do inference. Read this for more information.
* I have made some minor changes in 'predictor.py' and 'demo.py'.
* You should note that after Step 5, we have created some files('config.yml', 'metadata.json', 'output/model_final.pth') in the same directory as 'train.py'.
* We will need those files to do inference.
1. Inference on images
   - Run the following command in terminal. Remember to create the output directory before running.
      `python3 demo.py --input test/*.jpg --output detection_result/ --opts MODEL.WEIGHTS output/model_final.pth MODEL.DEVICE cuda`
      - 'test/*.jpg': is the directory where I put my test images.
      - 'detection_result/': is the directory to save the output images. You need to create this directory your own.
      - 'MODEL.WEIGHTS output/model_final.pth': is the path to our trained model. Normally is located at the folder 'output'.
      - 'MODEL.DEVICE cuda': You can choose either to use GPU or CPU. If CPU, it will be 'MODEL.DEVICE cpu'.
      - Below are some output images:
      ![image1](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/blob/main/resources/detection_result/res1.jpg)
      ![image2](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/blob/main/resources/detection_result/res2.jpg)
      ![image3](https://github.com/fookseng/Pytorch-Detectron2-object-detection-and-segmentation/blob/main/resources/detection_result/res3.jpg)
2. Inference on video
   - Run the following command in terminal.
   `python3 demo.py --video-input input.mp4 --output detection_result/ --opts MODEL.WEIGHTS output/model_final.pth `
4. Inference on webcam
   - I did not try this.

# Thank you, and GOOD LUCK :)
