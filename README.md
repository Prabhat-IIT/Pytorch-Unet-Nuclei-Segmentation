# Digital Pathology: Segmentation of Nuclei in Images

Grading and diagnosis of tumors in cancer patients have traditionally been done by examination of tissue specimens under a powerful microscope by expert pathologists. While this process continues to be widely applied in clinical settings, it is not scalable to translational and clinical research studies involving hundreds or thousands of tissue specimens. State-of-the-art digitizing microscopy instruments are capable of capturing high-resolution images of whole slide tissue specimens rapidly. Computer aided segmentation and classification has the potential to improve the tumor diagnosis and grading process as well as to enable quantitative studies of the mechanisms underlying disease onset and progression. We detect and segment all the nuclear material in a given set of image tiles extracted from whole slide tissue images by using modern deep learning approach.

## Getting Started

To clone the repository

`git clone https://github.com/SHIVA-sopho/Pytorch-Unet-Nuclei-Segmentation.git`

### Prerequisites

Basic dependencies stated in requirement.txt

To install dependencies: `pip install -r requirements.txt `

### Dataset
Dataset is taken from MICCAI Digital Pathology challenge.

[Download Dataset](http://quip2.bmi.stonybrook.edu:4000/segmentation_training_set.zip)

## Results
#### Training patches
![Nuclie patch](https://i.imgur.com/uisZx8s.png" "Nuclie patch")

#### Masks generated
![Mask](https://i.imgur.com/AuvbnSb.png "Mask patch")

#### Metrics and Scores on Validation Dataset
Mean_Dice =  0.8641389261779585

Mean_F1 =  0.7577543034961058

Mean_Aggr_Jacard =  0.6341042531927357
### Validation Split
The validation dataset is created using four images from 15 given training images. 

## Contributing

Please read [CONTRIBUTING.md](link here) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Shiva Gupta** - *network architecture* - [Shiva](https://github.com/SHIVA-sopho)
* **Sanjay Kumar** - *Defining metric and losses* - [Sanjay](https://github.com/sanjaykr5)
* **Prabhat Sharma** - *Dataset extraction* - [Prabhat](https://github.com/Prabhat-IIT)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* https://arxiv.org/pdf/1803.02786.pdf

