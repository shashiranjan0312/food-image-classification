from pathlib import Path

readme = r"""# 🍽️ Food Image Classification & Calorie Tracking

A computer-vision based food analysis project that combines **food image classification** with **calorie/meal tracking**. The project uses a trained Keras model to recognize food images and provides the foundation for estimating and recording meal information.

## ✨ Features

- 🖼️ **Food Image Classification**
  - Classifies food images using a trained deep-learning model.
  - Model is stored as `food_classifier_frozen_best.keras`.

- 🔥 **Calorie Tracking**
  - Includes a calorie-meter notebook for working with food/calorie information.
  - Supports the project goal of connecting food recognition with calorie monitoring.

- 📝 **Meal Logging**
  - The project includes meal-logging functionality.
  - Food and calorie information can be recorded as part of a user's meal history.

- 🧠 **Deep Learning Model**
  - Built using the Keras/TensorFlow ecosystem.
  - The trained model is kept separately from the application code for easier reuse.

- 📂 **Organized Project Structure**
  - Dataset-related files are maintained in `data/`.
  - Model-related files are maintained in `models/`.

## 🏗️ Project Structure

```text
Food-Image-Classification/
│
├── data/
│   └── Dataset / food-related data
│
├── models/
│   └── Model-related files
│
├── app.py
│   └── Main application
│
├── calorie_meter.ipynb
│   └── Calorie analysis / experimentation notebook
│
├── food_classifier_frozen_best.keras
│   └── Trained food classification model
│
├── .gitignore
├── .gitattributes
└── README.md
