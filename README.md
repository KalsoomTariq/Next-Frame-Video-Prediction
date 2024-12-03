# Deep Learning Next Frame Video Prediction

This project implements next-frame video prediction using deep learning models such as ConvLSTM, PredRNN, and Transformer. The goal is to predict future frames in a video sequence based on the previous frames. The project includes a Gradio-based frontend for interactive visualization and experimentation.

## Features
- **ConvLSTM**: Convolutional LSTM network for spatiotemporal sequence prediction.
- **PredRNN**: A state-of-the-art spatiotemporal model for video prediction.
- **Transformer**: Transformer architecture applied to video frame prediction.
- **Gradio Frontend**: A user-friendly interface for testing video prediction models interactively.
- **Support for Custom Datasets**: The project supports custom video datasets for frame prediction tasks.

  
### Prerequisites

Make sure that you have Python 3 Installed in your system

## Project Setup

### 1. Clone the Repository

1. Clone this repository to your local machine:

```bash
git clone https://github.com/KalsoomTariq/Next-Frame-Video-Prediction.git
cd Next-Frame-Video-Prediction
```


2. Create a virtual environment
```bash
python3 -m venv venv
```

3. Activate the virtual environment
```bash
source venv/bin/activate
```

4. Install the dependencies
```bash
pip install -r requirements.txt
```

5. Run server using command
```bash
python gradio_app.py
```

## Contributing

We welcome contributions to improve this project! If you have any suggestions, bug fixes, or new features, feel free to open an issue or submit a pull request. Please ensure that your contributions align with the overall goals of the project and that the code follows the existing code style.

### How to contribute:
1. Fork the repository to your own GitHub account.
2. Create a new branch for your feature or bug fix.
3. Make your changes and test them thoroughly.
4. Commit your changes with clear and concise commit messages.
5. Push your changes to your forked repository.
6. Open a pull request to the main repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
(TODO)

---
