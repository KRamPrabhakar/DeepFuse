This folder contains Lasagne code to generate results by DeepFuse model mentioned in our ICCV 2017 paper. 
Published by:
K Ram Prabhakar, Video Analytics Lab, Indian Insitute of Science, Bangalore, India on November 3, 2017.
Contact: ramprabhakar@iisc.ac.in
===================================================================================================
This code is released for research purposes only, not for commercial use. Please cite our paper if you use our code in your work:
@inproceedings{ram2017deepfuse,
  title={Deepfuse: A deep unsupervised approach for exposure fusion with extreme exposure image pairs},
  author={Ram Prabhakar, K and Sai Srikar, V and Venkatesh Babu, R},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={4714--4722},
  year={2017}
}
===================================================================================================
Software pre-requistes:
1. Theano 0.8.2
2. Lasagne 0.2.dev1
3. MATLAB R2015b (with IP and CV toolboxes)
===================================================================================================
Input processing:
1. Use Gen_Input.m to create data for DeepFuse
	a. Modify "input_fol" to locate the main folder with input images.
	b. Modify "savepath" to mention the location to store input to DeepFuse.

Steps to generate results by DeepFuse:
1. load_data.py - a python script to read input images.
	a. Modify "mypath_test_in" in line 13 to point to the location where input to DeepFuse is stored (should be same as "savepath" in "Gen_Input.m")
2. getres.py - a python script to run DeepFuse model.
	a. "Num_of_test_images" - modify this variable to indicate the number of test images 
	b. Modify line 20 to the location where you want to save the result
	c. After all modifications, run: "python getres.py" in terminal (should be run from the same folder location)
3. Above step will fuse luminance channel of input image pairs. To fuse color channels, run "Gen_Result.m" matlab script. 
	a. Modify "savepath" in line 2 to the path where the results has to be stored
	b. run the script
		i. Upon running the script, a matlab folder prompt will be displayed. Navigate and choose the underexposed image and overexposed image
		ii. The final result will be stored in the "savepath" location.
===================================================================================================
FAQ:
1. Can I train DeepFuse with this code?
	Ans: No, This is an inference code. This cannot be used to train DeepFuse. 
2. Can I run this code on CPU?
	Ans: yes, you can by setting device parameter in ~/.theanorc script present in your home folder. (from "device=gpu" to "device=cpu")
3. Can DeepFuse handle moving camera and moving object scenarios?
	Ans: No, DeepFuse is trained to fuse static scenes only. In case of camera motion, the input to DeepFuse has to be aligned using other softwares like PFSTOOLS.
===================================================================================================
