# Object Detection and Segmentation
# A Step-by-step guide to Pytorch Detectron2 with custom datasets
# Environments:
1. Google Colab (strongly recommended, ipynb file is provided.)
2. Ubuntu (what we will be using here.)

## For Google Colab:
Download or copy the ipynb file and run it on Google Colab.

## For Ubuntu:
### Step 1: Dealing with Anaconda
-Install Anaconda, since we will use Anaconda to manage environments.
[How to install Anaconda](https://docs.anaconda.com/anaconda/install/linux/)
-Open your terminal, and create a new Python environment with the name 'detectron2'(Any name should be fine).
~$ conda create -n detectron2
-Activate the environment we have created.
~$ conda activate detectron2
-If the above steps are successful, you will see something similar to this:
(detectron2) lemon@lemon-System-Product-Name:~$  * Note that "(detectron2)" has appeared. *

### Step 2: Install Detectron2
[How to install Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
#### Method 1: Which I was using (preferred).
1. ~$ cd Desktop (Change my current working directory to Desktop. You can change to any directory.)
2. ~$ git clone https://github.com/facebookresearch/detectron2.git
3. ~$ python3 -m pip install -e detectron2
4. ~$ ls (You will now see a folder called "detectron2" appeared in your Desktop.)
#### Method 2:
1. Check gcc version, since version >= 5.4 is needed.
   ~$ gcc --version
2. Check torch version and GPU availability.
   ~$ python3
   >>> import torch, torchvision
   >>> print(torch.__version__, torch.cuda.is_available())
   1.8.1+cu102 True
   >>> exit()
3. From [How to install Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), go to     "Install Pre-Built Detectron2 (Linux only)". Then, from the table, choose the correct installation. In my case, I select torch 1.8 and CUDA 10.2(These are what we saw in the above output.).
4. Copy the command and run it on your terminal.
   ~$ python3 python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html

### Step 3: Test our installation
1. Download a picture for testing purpose.
   ~$ wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
2. Run main.py 
   (Remember to comment line25-"cfg.MODEL.DEVICE='cpu'", if you do have a GPU.)
3. You can now see the predicted image(result.jpg) on your Desktop, like below:

### Step 4: Prepare our datasets
#### First: Label your images
We use the tool [labelme](https://github.com/wkentaro/labelme) to label our images.
An example for the image and annotation can be found here.
#### Second: Register your datasets
Since we must register our dataset in Detectron2 in order to use it[see here](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html).
There are few methods to do it.
##### Method 1: Convert annotations to COCO format dataset.(fast and simple, highly recommended.)
- A conversion tool is provided in my Github.
- After you have done the conversion, we are able to use the function 'register_coco_instances' to register our datasets easily.
##### Method 2: Write your own function to register the dataset.(Not tested in this tutorial.)
If you are using labelme, a sample is provided for your reference(register_dataset.py).
Credit to [Jadezzz])(https://github.com/Jadezzz)
#### Final: Visualize the images and annotations to check if there is any error.

### Step 5: Let's start to train!


 

Use Detectron2 to do object detection and segmentation on custom dataset.
