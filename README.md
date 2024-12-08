# gan_creative_models
README: Monet-Style Image Generation Using CycleGAN
Project Title
Monet-Style Image Generation Using CycleGAN

Project Overview
This project leverages CycleGAN, a type of Generative Adversarial Network (GAN), to transform real-world photographs into Monet-style paintings. CycleGAN enables unpaired image-to-image translation, making it ideal for scenarios where paired datasets are unavailable. By training on real-world photos and Monet paintings, this project demonstrates the application of AI in automating artistic style transfer while preserving structural content.

Features
Unpaired Style Transfer: Generate Monet-style paintings without paired datasets.
Cycle Consistency: Ensures that transformed images can be reverted to their original form.
Preserves Content: Maintains the structural integrity of the original images.
Customizable Framework: Easily adaptable to other artistic styles like Van Gogh or Picasso.
Folder Structure
bash
Copy code
Project Root
├── data/
│   ├── monet_jpg/        # Contains Monet-style paintings
│   ├── photo_jpg/        # Contains real-world photographs
├── models/
│   ├── generator_g/      # Photo to Monet generator
│   ├── generator_f/      # Monet to photo generator
│   ├── discriminator_x/  # Discriminator for Monet-style images
│   ├── discriminator_y/  # Discriminator for real-world photos
├── scripts/
│   ├── preprocessing.py  # Code for data preprocessing
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation metrics
├── results/
│   ├── generated_images/ # Generated Monet-style paintings
│   ├── comparisons/      # Side-by-side comparisons
├── README.md             # Project details
└── requirements.txt      # Python dependencies
Requirements
Python 3.8 or later
TensorFlow 2.8 or later
Matplotlib
NumPy
OpenCV
GPU (Recommended for faster training)
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Dataset
Monet Paintings: 300 images sourced from publicly available datasets.
Real Photos: 7,028 landscape images representing diverse environments.
Preprocessed to 256×256 resolution, normalized to 
[
−
1
,
1
]
[−1,1], and augmented with random flips and rotations.
Training the Model
Preprocess the Dataset: Run the preprocessing.py script to resize, normalize, and batch the images.

bash
Copy code
python scripts/preprocessing.py
Train the Model: Execute the train.py script to train the CycleGAN model. Default training runs for 20 epochs.

bash
Copy code
python scripts/train.py
Save Checkpoints: Model checkpoints are automatically saved every 5 epochs in the models/ directory.

Evaluation
Quantitative Metrics:

Fréchet Inception Distance (FID): Measures the similarity between distributions of real and generated images.
Inception Score (IS): Evaluates the quality and diversity of the generated images.
Visual Comparison: Run the evaluate.py script to generate side-by-side comparisons of input photos and Monet-style outputs.

bash
Copy code
python scripts/evaluate.py
Results
Metrics:
Fréchet Inception Distance (FID): 28.5
Inception Score (IS): 4.3
Qualitative Output: The generated Monet-style paintings replicate the vibrant colors and brushstroke patterns of Monet while preserving the content of the original photos.
Applications
Art and Media: Automate artistic content creation for personal or commercial use.
Education: Teach artistic styles using AI-generated examples.
Real-Time Video: Potential to expand to real-time video style transfer.
Future Work
Optimize for real-time style transfer.
Expand to include other artistic styles.
Improve model generalization for diverse input domains.
Integrate into a user-friendly web or mobile application.
References
Zhu, J.-Y., et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." arXiv preprint arXiv:1703.10593, 2017.
Goodfellow, I., et al. "Generative Adversarial Networks." Communications of the ACM, 2014.
TensorFlow Documentation: https://www.tensorflow.org/tutorials/generative/cyclegan
