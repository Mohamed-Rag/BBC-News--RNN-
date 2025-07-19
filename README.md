# BBC News Classification using RNN

## Overview
This project implements a Recurrent Neural Network (RNN) model to classify BBC news articles into one of six categories. Using deep learning techniques with TensorFlow and Keras, the model analyzes news article text and automatically categorizes them based on their content.

## Project Description
The BBC News RNN Classifier is a text classification system that demonstrates the application of recurrent neural networks for natural language processing tasks. The model uses bidirectional LSTM layers to understand the sequential nature of text and classify news articles into predefined categories.

## Dataset
The project uses the `BBC News.csv` dataset which contains:
- **News Articles**: Text content of BBC news articles
- **Categories**: Six different news categories for classification
- **Training Data**: 1,780 articles for training
- **Validation Data**: 445 articles for validation

## Model Architecture
The RNN model consists of:
- **Embedding Layer**: Converts words to dense vectors (vocab_size=1000, embedding_dim=16)
- **Bidirectional LSTM**: Processes text sequences in both directions (16 units)
- **Dense Layer**: Fully connected layer with ReLU activation (24 units)
- **Output Layer**: Softmax activation for 6-class classification

### Model Summary
```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
embedding (Embedding)       (None, 120, 16)           16000     
bidirectional (Bidirectional) (None, 32)              2176      
dense (Dense)               (None, 24)                792       
dense_1 (Dense)             (None, 6)                 150       
=================================================================
Total params: 19,118 (74.68 KB)
Trainable params: 19,118 (74.68 KB)
Non-trainable params: 0 (0.00 Byte)
```

## Features
- **Text Preprocessing**: Tokenization and sequence padding
- **RNN Classification**: Bidirectional LSTM for sequence processing
- **Multi-class Classification**: Categorizes into 6 news categories
- **Performance Visualization**: Training and validation metrics plotting
- **High Accuracy**: Achieves 91.91% validation accuracy

## Technology Stack
- **Python**: Primary programming language
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **CSV**: Data handling

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- Jupyter Notebook
- Required Python libraries

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mohamed-Rag/BBC-News--RNN-.git
   cd BBC-News--RNN-
   ```

2. **Install required dependencies**:
   ```bash
   pip install tensorflow numpy matplotlib pandas scikit-learn
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Open the project notebook**:
   Open `project1.ipynb` in your Jupyter environment

## Usage

### Running the Classification Model

1. **Data Loading**: The notebook loads the BBC News dataset from `BBC News.csv`
2. **Preprocessing**: Text tokenization, sequence padding, and label encoding
3. **Model Training**: Train the RNN model for 10 epochs
4. **Evaluation**: Assess model performance on validation data
5. **Visualization**: Plot training and validation metrics

### Model Parameters
```python
vocab_size = 1000           # Vocabulary size
embedding_dim = 16          # Embedding dimension
max_length = 120           # Maximum sequence length
training_portion = 0.8     # 80% training, 20% validation
```

## Project Structure
```
BBC-News--RNN-/
├── BBC News.csv           # Dataset file
├── project1.ipynb         # Main Jupyter notebook with RNN implementation
└── README.md             # This file
```

## Model Performance
The trained model achieves:
- **Training Accuracy**: 98.99% (final epoch)
- **Validation Accuracy**: 91.91%
- **Training Loss**: 0.0638 (final epoch)
- **Validation Loss**: 0.2760

### Training Progress
The model shows consistent improvement across epochs:
- Epoch 1: 22.87% accuracy
- Epoch 5: 91.69% accuracy
- Epoch 10: 98.99% accuracy

## Text Preprocessing
The preprocessing pipeline includes:
- **Tokenization**: Converting text to sequences of integers
- **Vocabulary Building**: Creating word-to-index mappings
- **Sequence Padding**: Ensuring uniform input length
- **Label Encoding**: Converting category labels to numerical format

## Model Training Details
- **Optimizer**: Adam optimizer
- **Loss Function**: Sparse categorical crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Batch Size**: Default (32)

## Visualization
The project includes visualization of:
- Training vs. validation accuracy curves
- Training vs. validation loss curves
- Model performance metrics over epochs

## Applications
This model can be used for:
- **Automated News Categorization**: Classify incoming news articles
- **Content Management**: Organize news content by category
- **Information Retrieval**: Improve search and filtering systems
- **Media Analysis**: Analyze news distribution across categories

## Limitations
- **Dataset Size**: Limited to BBC news articles
- **Category Scope**: Only six predefined categories
- **Language**: English language only
- **Domain Specific**: Trained specifically on news content

## Future Improvements
- Expand to more news categories
- Implement attention mechanisms
- Add support for multiple languages
- Include confidence scores for predictions
- Implement real-time classification API

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## Educational Value
This project demonstrates:
- RNN implementation for text classification
- Deep learning with TensorFlow/Keras
- Natural language processing techniques
- Model evaluation and visualization
- Practical application of neural networks

## License
This project is open-source and available for educational and research purposes.

## Contact
For questions, suggestions, or collaboration opportunities, please contact Mohamed-Rag through GitHub.

## Acknowledgments
- BBC for providing the news dataset
- TensorFlow team for the deep learning framework
- The open-source community for supporting tools and libraries

