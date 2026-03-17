---
layout: default
title: Linear Algebra Visualizer (2026)
---

## Linear Algebra Visualizer (2026)

### An Interactive Visualization Platform for Matrix Transformations, SVD, PCA, and Least Squares (LSE)

Linear algebra underlies modern machine learning, computer vision, medical imaging, and scientific computing. However, students often experience matrices as abstract symbolic procedures rather than geometric transformations.

**Linear Algebra Visualizer** is an interactive Streamlit based visualization platform designed to bridge algebraic computation and geometric intuition. Instead of treating matrices as static arrays of numbers, this system demonstrates how matrices act as geometric operators that rotate, scale, project, compress, and transform data in real time.

---

## 🚀 Live Demo

👉 **[Open Interactive Linear Algebra Visualizer](https://visualizer2026-consm2v7nnfpbvynupexca.streamlit.app)**  
📎 **[Short Link](http://bit.ly/Visualizer2026)**  

---

## 🔗 Project Access

- **[GitHub Repository](https://github.com/BM-Studio-2026/Visualizer2026/)**  
- **[Project Homepage (GitHub Pages)](https://bm-studio-2026.github.io/Visualizer2026/)**  

---

## Features

### 2D Linear Transformations
- Real time visualization of matrix actions on vectors and grids  
- Rotation, scaling, shear, and custom matrices  

### 3D Transformations
- Interactive 3D geometric transformations  
- Visualization of basis changes and spatial mapping  

### Projection and Lifting
- 2×3 projection (dimension reduction)  
- 3×2 lifting (dimension expansion)  
- Geometric interpretation of linear mappings between spaces  

### Singular Value Decomposition (SVD)
- Visual breakdown of a matrix transformation into:
  - Rotation (Vᵀ)  
  - Scaling (Σ)  
  - Rotation (U)  
- Interactive control of singular values  
- Demonstration of rank and conditioning  

### Principal Component Analysis (PCA)
- Variance visualization  
- Major and minor axis interpretation  
- Covariance geometry explanation  

### Least Squares Estimation (LSE)
- Interactive demo of fitting an overdetermined system **Ax ≈ b**  
- Visual intuition for noisy data and approximation  
- Geometric meaning as projection:
  - **b projected onto column space of A**  
  - residual **r = b − Ax** orthogonal to column space  
- Optional concepts:
  - **AᵀA x = Aᵀ b**  
  - connection to pseudoinverse and SVD  

### Image Compression
- SVD based image compression  
- PCA based image compression  
- Rank reduction demonstrations  
- Visual comparison of compression levels  

---

## Educational Goals

This platform is designed to help students:

- Develop geometric intuition for matrix operations  
- Understand the meaning of singular value decomposition (SVD)  
- Visualize rank deficiency and ill conditioning  
- Connect PCA to variance and covariance structure  
- Understand least squares as projection and error minimization  
- See dimensionality reduction and fitting as geometry  

The goal is not merely to compute linear algebra but to make it visible.

---

## Why This Project Matters

Modern AI systems rely heavily on linear algebra. From neural networks to medical imaging reconstruction, matrix operations drive nearly every computation. Yet many students struggle to build intuitive understanding of these concepts.

By integrating 2D and 3D transformations, SVD, PCA, least squares estimation, and image compression into a unified interactive environment, this project transforms linear algebra from symbolic manipulation into geometric experience.

---

## Technologies Used

- Python  
- Streamlit  
- NumPy  
- Matplotlib or Plotly  
- PIL (Image Processing)  
- Manim (Mathematical Animation Engine, originally developed by 3Blue1Brown)  

---

## Project Structure

```
Visualizer2026/
│
├── Home.py              # Main application entry
├── pages/               # Multi-page interactive modules
├── assets/              # Images and visual assets
├── requirements.txt     # Dependencies
├── README.md            # Repository overview
└── index.md             # GitHub Pages homepage
```

---

## Future Development

- Educational lesson integration  
- Additional machine learning algorithm demonstrations  

---

## Author

**Brayden Miao**  
BM Studio · High School Research Project (2026)

---

## License

This project is open for educational use.
