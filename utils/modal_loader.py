'''
    DEEP LEARNING PROJECT FOR NEXT FRAME VIDEO PREDICTION

Members:
- Kalsoom Tariq (i212487)
- Abtaal Aatif (i212990)
- Ali Ashraf (i210756))

'''

import torch
import torch.nn as nn
import numpy as np
import numpy as np  # For array manipulations
import tensorflow as tf  # For loading and running the PredRNN model

class ModelLoader:
    def __init__(self):
        # Dictionary to store model classes
        self.models = {
            'ConvLSTM': self.load_convlstm,
            'PredRNN': self.load_predrnn,
            'Transformer': self.load_transformer
        }
    
    def load_convlstm(self):
        """
        Load ConvLSTM model with pre-trained weights
        
        Returns:
            nn.Module: Loaded ConvLSTM model
        """
        class ConvLSTMCell(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size):
                super(ConvLSTMCell, self).__init__()
                self.activation = torch.tanh if activation == "tanh" else torch.relu
                self.conv = nn.Conv2d(
                    in_channels=in_channels + out_channels,
                    out_channels=4 * out_channels,
                    kernel_size=kernel_size,
                    padding=padding
                )
                self.W_ci = nn.Parameter(torch.zeros(out_channels, *frame_size))
                self.W_co = nn.Parameter(torch.zeros(out_channels, *frame_size))
                self.W_cf = nn.Parameter(torch.zeros(out_channels, *frame_size))
            
            def forward(self, X, H_prev, C_prev):
                if H_prev is None:
                    H_prev = torch.zeros(X.size(0), self.conv.out_channels // 4, X.size(2), X.size(3), device=X.device)
                if C_prev is None:
                    C_prev = torch.zeros_like(H_prev)
                conv_output = self.conv(torch.cat([X, H_prev], dim=1))
                i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)
                input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
                forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)
                C = forget_gate * C_prev + input_gate * self.activation(C_conv)
                output_gate = torch.sigmoid(o_conv + self.W_co * C)
                H = output_gate * self.activation(C)
                return H, C

        class ConvLSTM(nn.Module):
            def __init__(self, num_channels, num_kernels, kernel_size, padding, activation, frame_size, num_layers):
                super(ConvLSTM, self).__init__()
                self.layers = nn.ModuleList()
                for i in range(num_layers):
                    in_ch = num_channels if i == 0 else num_kernels
                    self.layers.append(ConvLSTMCell(in_ch, num_kernels, kernel_size, padding, activation, frame_size))
                self.final_conv = nn.Conv2d(
                    in_channels=num_kernels,
                    out_channels=num_channels,
                    kernel_size=(3, 3),
                    padding=(1, 1)
                )
            
            def forward(self, X):
                batch_size, seq_len, _, height, width = X.size()
                H, C = [None] * len(self.layers), [None] * len(self.layers)
                outputs = []
                for t in range(seq_len):  # Process each time step
                    frame = X[:, t]
                    for l, layer in enumerate(self.layers):
                        H[l], C[l] = layer(frame, H[l], C[l])
                        frame = H[l]
                    if t >= seq_len - 5:  # Collect outputs for the last 5 time steps
                        outputs.append(self.final_conv(H[-1]))
                return torch.stack(outputs, dim=1)  # Shape: (batch_size, 5, channels, height, width)

        # Instantiate model
        model = ConvLSTM(
            num_channels=1,
            num_kernels=64,
            kernel_size=(3, 3),
            padding=(1, 1),
            activation="relu",
            frame_size=(64, 64),
            num_layers=3
        )
        
        weights_path = 'weights/lstm_best_model.pth'
        try:
            checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
            # Check and remove 'module.' prefix if needed
            if any(key.startswith('module.') for key in checkpoint.keys()):
                checkpoint = {key.replace('module.', ''): val for key, val in checkpoint.items()}
            
            # Load the updated state dictionary
            model.load_state_dict(checkpoint)
            print(f"Weights loaded from {weights_path}")
        except FileNotFoundError:
            print(f"Pre-trained weights not found at {weights_path}, using randomly initialized model.")
        
        return model

    def load_transformer(self):
        """
        Load Transformer model with pre-trained weights
        
        Returns:
            nn.Module: Loaded Transformer model
        """
        class Transformer(nn.Module):
            def __init__(self, num_channels, embed_dim, nhead, num_layers, frame_size):
                super(Transformer, self).__init__()
                self.embedding = nn.Conv2d(
                    in_channels=num_channels, 
                    out_channels=embed_dim, 
                    kernel_size=3, 
                    padding=1
                )
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead),
                    num_layers=num_layers
                )
                self.decoder = nn.Conv2d(
                    in_channels=embed_dim, 
                    out_channels=num_channels, 
                    kernel_size=3, 
                    padding=1
                )
                self.frame_size = frame_size
            
            def forward(self, X):
                batch_size, seq_len, _, height, width = X.size()
                X = X.view(batch_size * seq_len, 1, height, width)  # Flatten time dimension
                embeddings = self.embedding(X).view(batch_size, seq_len, -1)  # (B, T, F)
                embeddings = embeddings.permute(1, 0, 2)  # (T, B, F) for Transformer
                encoded = self.encoder(embeddings)
                decoded = encoded.permute(1, 0, 2).view(batch_size, seq_len, -1, height, width)
                outputs = [self.decoder(decoded[:, t]) for t in range(seq_len - 5, seq_len)]
                return torch.stack(outputs, dim=1)

        # Instantiate model
        model = Transformer(
            num_channels=1,
            embed_dim=64,
            nhead=4,
            num_layers=2,
            frame_size=(64, 64)
        )
        
        # Load pre-trained weights
        weights_path = 'path_to_transformer_weights.pth'
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            print(f"Weights loaded from {weights_path}")
        except FileNotFoundError:
            print(f"Pre-trained weights not found at {weights_path}, using randomly initialized model.")
        
        return model
        
    def predict(self, model_name, input_video):
        """
        Generate video prediction
        
        Args:
            model_name (str): Name of the model to use
            input_video (torch.Tensor): Input video tensor
        
        Returns:
            np.ndarray: Predicted video frames
        """
        # Load the specified model
        model = self.models[model_name]()
        model.eval()
        
        # Perform prediction
        with torch.no_grad():
            predictions = model(input_video)
        
        # Convert to numpy for visualization
        predictions = predictions.permute(0, 1, 3, 4, 2).numpy()
        predictions = (predictions * 255).astype(np.uint8)
        
        return predictions
    


class PredRNNModel:
    def __init__(self, model_path):
        """
        Initialize the PredRNNModel by loading the pre-trained model weights.
        """
        self.model = self.load_predrnn_model(model_path)

    def load_predrnn_model(self, model_path):
        """
        Load the PredRNN model with pre-trained weights.

        Args:
            model_path (str): Path to the saved model.
        
        Returns:
            tf.keras.Model: Loaded PredRNN model.
        """
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"PredRNN model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading PredRNN model: {e}")
            raise e

    def predict(self, input_frames):
        """
        Predict the next frames using the PredRNN model.

        Args:
            input_frames (np.ndarray): Input frames with shape (batch_size, seq_length, height, width, channels).

        Returns:
            np.ndarray: Predicted frames with shape (batch_size, prediction_length, height, width, channels).
        """
        return self.model.predict(input_frames)