# Medical Imaging Lab Work

This repository contains practical laboratory work completed in the field of medical imaging. The projects cover image segmentation, tomographic reconstruction, scintigraphic image reconstruction, and multimodal image fusion with segmentation.

## **Project Overview**

The purpose of this lab work was to explore important concepts and techniques used in medical imaging through a series of practical sessions.

The work includes:
- Tumor segmentation using 3D Slicer
- Tomographic image reconstruction
- Cardiac scintigraphic image reconstruction
- Multimodal MRI and ultrasound fusion with automatic segmentation

## **Laboratory Work**

### **TP1: Tumor Segmentation with 3D Slicer**

This practical work focused on tumor segmentation using **3D Slicer**.

The following approaches were studied:
- **Manual segmentation**
- **Semi-automatic segmentation**
- **Automatic segmentation**

This TP introduced the use of segmentation tools in medical imaging and allowed comparison between different segmentation strategies.

### **TP2: Tomographic Image Reconstruction**

This practical work focused on reconstructing tomographic images in computed tomography (**CT**).

The main tasks included:
- Loading and displaying the **Shepp-Logan phantom**
- Computing the **sinogram** using the **Radon transform**
- Reconstructing the image using **Filtered Back Projection (FBP)**
- Reconstructing the image using **SART**
- Comparing both methods using metrics such as **MSE**, **PSNR**, and **SSIM**

This TP helped analyze reconstruction quality and understand the differences between classical reconstruction methods.

### **TP3: Cardiac Scintigraphic Image Reconstruction**

This practical work explored the simulation and reconstruction of a cardiac scintigraphic image.

The main tasks included:
- Creating a cardiac phantom
- Adding **Poisson noise**
- Computing the sinogram
- Reconstructing the image using **FBP** and **SART**
- Applying denoising
- Detecting cold regions using intensity thresholding
- Evaluating the results using **MSE**, **SSIM**, **PSNR**, **MAE**, and relative error

This TP focused on image reconstruction, noise handling, and analysis of pathological regions in nuclear imaging.

### **TP4: Multimodal Reconstruction and Fusion**

This practical work focused on multimodal medical image simulation, reconstruction, fusion, and segmentation.

The main tasks included:
- Generating the **Shepp-Logan phantom**
- Simulating **MRI** acquisition and reconstruction
- Simulating **ultrasound** imaging with speckle noise
- Applying denoising methods
- Performing multimodal fusion using wavelets and weighted averaging
- Applying **Otsu automatic segmentation**
- Comparing results using **Dice** and **IoU**

This TP highlighted the importance of combining complementary imaging modalities to improve image interpretation and segmentation quality.

## **Objectives**

The main objectives of this repository are:
- To understand key concepts in medical imaging
- To apply image processing and reconstruction techniques
- To evaluate different reconstruction and segmentation methods
- To explore multimodal fusion in medical image analysis

## **Tools and Technologies**

- **3D Slicer**
- **Python**
- **NumPy**
- **Matplotlib**
- **Scikit-image**
- **SciPy**
- **PyWavelets**

## **Repository Structure**

```bash
.
├── TP1/
├── TP2/
├── TP3/
├── TP4/
├── projet/
├── .gitignore/
└── README.md
```

## **How to Use**

1. Open each TP folder to access the corresponding practical work.
2. Review the scripts, datasets, and outputs.
3. Run the code in Python or open the segmentation work in 3D Slicer depending on the TP.
4. Compare the obtained results and evaluation metrics.

## **Academic Context**

This repository was created as part of university laboratory work in **medical imaging**. It gathers different practical sessions that illustrate segmentation, reconstruction, denoising, fusion, and evaluation methods used in medical image analysis.

## **Notes**

This work is intended for academic and educational purposes only.
