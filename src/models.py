import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple2DCNN(nn.Module):
    """
    Simple 2D CNN for deforestation detection.
    Takes a single satellite image as input.
    """
    def __init__(self, in_channels=6, out_channels=1):
        super(Simple2DCNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # First upsampling block
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Second upsampling block
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Third upsampling block
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final convolution to get output
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, out_channels, height, width]
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Simple3DCNN(nn.Module):
    """
    3D CNN for spatio-temporal deforestation detection.
    Takes a sequence of satellite images over time.
    """
    def __init__(self, in_channels=6, time_steps=5, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # Don't pool along time dimension yet
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))   # Now pool time as well
        )
        
        # Calculate output size after encoder
        # Time dimension is reduced by 2x, spatial dimensions by 4x each
        self.time_factor = time_steps // 2
        
        self.decoder = nn.Sequential(
            # Upsample spatial dimensions but not time
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ReLU(),
            
            # Collapse time dimension and upsample spatial again
            nn.Conv3d(32, 16, kernel_size=(self.time_factor, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, channels, time_steps, height, width]
            
        Returns:
            Output tensor of shape [batch_size, out_channels, height, width]
        """
        # Encoder: 3D convolutions
        x = self.encoder(x)
        
        # Reshape for 2D convolutions - collapse time dimension
        batch_size, channels, t, h, w = x.shape
        x = x.view(batch_size, channels * t, h, w)
        
        # Final convolution to get output
        x = self.decoder(x)
        return x


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for spatio-temporal modeling
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Gates for input, forget, output, and cell update
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,  # 4 gates: input, forget, cell, output
            kernel_size=kernel_size,
            padding=padding
        )
        
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for a single time step
        
        Args:
            x: Input tensor [batch, channels, height, width]
            h_prev: Previous hidden state [batch, hidden_channels, height, width]
            c_prev: Previous cell state [batch, hidden_channels, height, width]
            
        Returns:
            h_next: Next hidden state
            c_next: Next cell state
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Calculate gates
        gates = self.gates(combined)
        
        # Split into separate gates
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)
        
        # Apply activations
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        
        # Update cell state
        c_next = forget_gate * c_prev + input_gate * cell_gate
        
        # Update hidden state
        h_next = output_gate * torch.tanh(c_next)
        
        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM for spatio-temporal deforestation detection.
    Processes a sequence of satellite images over time.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3, num_layers=2, output_channels=1):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        # Create ConvLSTM layers
        self.convlstm_cells = nn.ModuleList()
        
        # First layer takes input data
        self.convlstm_cells.append(ConvLSTMCell(
            input_channels, hidden_channels, kernel_size
        ))
        
        # Subsequent layers take output from previous layer
        for _ in range(1, num_layers):
            self.convlstm_cells.append(ConvLSTMCell(
                hidden_channels, hidden_channels, kernel_size
            ))
        
        # Output layer for final prediction
        self.output_layer = nn.Sequential(
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass through all time steps and layers
        
        Args:
            x: Input tensor [batch, time_steps, channels, height, width]
            
        Returns:
            Output tensor [batch, output_channels, height, width]
        """
        batch_size, time_steps, _, height, width = x.shape
        
        # Initialize hidden and cell states for all layers
        h = []
        c = []
        for i in range(self.num_layers):
            h.append(torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device))
            c.append(torch.zeros(batch_size, self.hidden_channels, height, width).to(x.device))
        
        # Process each time step
        for t in range(time_steps):
            # Get current input at this time step
            current_input = x[:, t, :, :, :]
            
            # Pass through each ConvLSTM layer
            for layer_idx in range(self.num_layers):
                # First layer takes the input, other layers take previous layer's output
                if layer_idx == 0:
                    input_tensor = current_input
                else:
                    input_tensor = h[layer_idx-1]
                
                # Update hidden and cell states
                h[layer_idx], c[layer_idx] = self.convlstm_cells[layer_idx](
                    input_tensor, h[layer_idx], c[layer_idx]
                )
        
        # Use final hidden state from top layer for prediction
        output = self.output_layer(h[-1])
        return output


class UNet(nn.Module):
    """
    UNet architecture for segmentation tasks.
    """
    def __init__(self, in_channels, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._encoder_block(in_channels, 64)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256)
        self.enc4 = self._encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling)
        self.dec1 = self._decoder_block(1024, 512)
        self.dec2 = self._decoder_block(512, 256)
        self.dec3 = self._decoder_block(256, 128)
        self.dec4 = self._decoder_block(128, 64)
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1[:-1](x)  # Skip the pooling
        enc1_pool = nn.MaxPool2d(kernel_size=2, stride=2)(enc1)
        
        enc2 = self.enc2[:-1](enc1_pool)  # Skip the pooling
        enc2_pool = nn.MaxPool2d(kernel_size=2, stride=2)(enc2)
        
        enc3 = self.enc3[:-1](enc2_pool)  # Skip the pooling
        enc3_pool = nn.MaxPool2d(kernel_size=2, stride=2)(enc3)
        
        enc4 = self.enc4[:-1](enc3_pool)  # Skip the pooling
        enc4_pool = nn.MaxPool2d(kernel_size=2, stride=2)(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4_pool)
        
        # Decoder with skip connections
        dec1 = self.dec1[0](bottleneck)  # ConvTranspose
        dec1 = torch.cat([dec1, enc4], dim=1)  # Skip connection
        dec1 = self.dec1[1:](dec1)  # Conv blocks
        
        dec2 = self.dec2[0](dec1)  # ConvTranspose
        dec2 = torch.cat([dec2, enc3], dim=1)  # Skip connection
        dec2 = self.dec2[1:](dec2)  # Conv blocks
        
        dec3 = self.dec3[0](dec2)  # ConvTranspose
        dec3 = torch.cat([dec3, enc2], dim=1)  # Skip connection
        dec3 = self.dec3[1:](dec3)  # Conv blocks
        
        dec4 = self.dec4[0](dec3)  # ConvTranspose
        dec4 = torch.cat([dec4, enc1], dim=1)  # Skip connection
        dec4 = self.dec4[1:](dec4)  # Conv blocks
        
        # Final output
        output = self.final(dec4)
        
        return output